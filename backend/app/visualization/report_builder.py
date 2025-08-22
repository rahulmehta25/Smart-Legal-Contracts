"""
Drag-and-Drop Report Builder

Provides intuitive drag-and-drop interface for creating custom reports with
visual elements, data connections, and advanced layout management.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from fastapi import WebSocket
import redis

from .charts import ChartEngine
from .reports import ReportBuilder


class ElementType(Enum):
    """Types of report elements"""
    TEXT = "text"
    CHART = "chart"
    TABLE = "table"
    IMAGE = "image"
    KPI = "kpi"
    FILTER = "filter"
    SPACER = "spacer"
    CONTAINER = "container"
    HEADER = "header"
    FOOTER = "footer"


class LayoutType(Enum):
    """Layout types for containers"""
    GRID = "grid"
    FLEX = "flex"
    ABSOLUTE = "absolute"
    FLOW = "flow"


class DataConnectionType(Enum):
    """Data connection types"""
    SQL_QUERY = "sql_query"
    API_ENDPOINT = "api_endpoint"
    FILE_UPLOAD = "file_upload"
    LIVE_STREAM = "live_stream"
    STATIC_DATA = "static_data"


@dataclass
class Position:
    """Element position"""
    x: float
    y: float
    width: float
    height: float
    z_index: int = 0


@dataclass
class Style:
    """Element styling"""
    background_color: Optional[str] = None
    border_color: Optional[str] = None
    border_width: Optional[int] = None
    border_radius: Optional[int] = None
    padding: Optional[int] = None
    margin: Optional[int] = None
    font_family: Optional[str] = None
    font_size: Optional[int] = None
    font_weight: Optional[str] = None
    color: Optional[str] = None
    text_align: Optional[str] = None
    opacity: Optional[float] = None
    shadow: Optional[str] = None


@dataclass
class DataConnection:
    """Data connection configuration"""
    id: str
    type: DataConnectionType
    config: Dict[str, Any]
    refresh_interval: Optional[int] = None
    cache_duration: Optional[int] = None
    last_updated: Optional[datetime] = None


@dataclass
class ReportElement:
    """Report builder element"""
    id: str
    type: ElementType
    position: Position
    style: Style
    content: Dict[str, Any]
    data_connection: Optional[DataConnection] = None
    parent_id: Optional[str] = None
    children: List[str] = None
    locked: bool = False
    visible: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.children is None:
            self.children = []


@dataclass
class ReportCanvas:
    """Report builder canvas"""
    id: str
    name: str
    description: str
    elements: List[ReportElement]
    layout: LayoutType
    canvas_size: Dict[str, float]
    grid_size: int = 10
    snap_to_grid: bool = True
    created_by: str = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


@dataclass
class ElementTemplate:
    """Template for report elements"""
    id: str
    name: str
    type: ElementType
    preview_image: Optional[str]
    default_config: Dict[str, Any]
    description: str
    category: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DragDropReportBuilder:
    """
    Advanced drag-and-drop report builder
    """
    
    def __init__(self, 
                 chart_engine: ChartEngine,
                 report_builder: ReportBuilder,
                 redis_client: redis.Redis):
        self.chart_engine = chart_engine
        self.report_builder = report_builder
        self.redis_client = redis_client
        
        # Storage
        self.canvases: Dict[str, ReportCanvas] = {}
        self.templates: Dict[str, ElementTemplate] = {}
        self.data_connections: Dict[str, DataConnection] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, List[WebSocket]] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
    
    async def create_canvas(self, 
                           name: str,
                           description: str,
                           layout: LayoutType = LayoutType.GRID,
                           canvas_size: Dict[str, float] = None,
                           created_by: str = None) -> str:
        """
        Create new report canvas
        
        Args:
            name: Canvas name
            description: Canvas description
            layout: Layout type
            canvas_size: Canvas dimensions
            created_by: Creator user ID
        
        Returns:
            Canvas ID
        """
        canvas_id = str(uuid.uuid4())
        
        if canvas_size is None:
            canvas_size = {"width": 1200, "height": 800}
        
        canvas = ReportCanvas(
            id=canvas_id,
            name=name,
            description=description,
            elements=[],
            layout=layout,
            canvas_size=canvas_size,
            created_by=created_by
        )
        
        self.canvases[canvas_id] = canvas
        
        # Cache canvas
        await self._cache_canvas(canvas_id, canvas)
        
        return canvas_id
    
    async def add_element(self, 
                         canvas_id: str,
                         element_type: ElementType,
                         position: Position,
                         content: Dict[str, Any],
                         style: Optional[Style] = None,
                         data_connection_id: Optional[str] = None) -> str:
        """
        Add element to canvas
        
        Args:
            canvas_id: Canvas identifier
            element_type: Type of element
            position: Element position
            content: Element content configuration
            style: Optional styling
            data_connection_id: Optional data connection
        
        Returns:
            Element ID
        """
        canvas = await self.get_canvas(canvas_id)
        if not canvas:
            raise ValueError(f"Canvas {canvas_id} not found")
        
        element_id = str(uuid.uuid4())
        
        # Get data connection if specified
        data_connection = None
        if data_connection_id:
            data_connection = self.data_connections.get(data_connection_id)
        
        # Create element
        element = ReportElement(
            id=element_id,
            type=element_type,
            position=position,
            style=style or Style(),
            content=content,
            data_connection=data_connection
        )
        
        # Snap to grid if enabled
        if canvas.snap_to_grid:
            element.position = self._snap_to_grid(element.position, canvas.grid_size)
        
        # Add to canvas
        canvas.elements.append(element)
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_added(canvas_id, element)
        
        return element_id
    
    async def move_element(self, 
                          canvas_id: str,
                          element_id: str,
                          new_position: Position) -> ReportElement:
        """
        Move element to new position
        
        Args:
            canvas_id: Canvas identifier
            element_id: Element identifier
            new_position: New position
        
        Returns:
            Updated element
        """
        canvas = await self.get_canvas(canvas_id)
        element = self._find_element(canvas, element_id)
        
        if not element:
            raise ValueError(f"Element {element_id} not found")
        
        if element.locked:
            raise ValueError(f"Element {element_id} is locked")
        
        # Check for collisions
        if await self._check_collision(canvas, element_id, new_position):
            new_position = await self._resolve_collision(canvas, element_id, new_position)
        
        # Snap to grid if enabled
        if canvas.snap_to_grid:
            new_position = self._snap_to_grid(new_position, canvas.grid_size)
        
        # Update position
        element.position = new_position
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_moved(canvas_id, element)
        
        return element
    
    async def resize_element(self, 
                            canvas_id: str,
                            element_id: str,
                            new_size: Dict[str, float]) -> ReportElement:
        """
        Resize element
        
        Args:
            canvas_id: Canvas identifier
            element_id: Element identifier
            new_size: New size (width, height)
        
        Returns:
            Updated element
        """
        canvas = await self.get_canvas(canvas_id)
        element = self._find_element(canvas, element_id)
        
        if not element:
            raise ValueError(f"Element {element_id} not found")
        
        if element.locked:
            raise ValueError(f"Element {element_id} is locked")
        
        # Update size
        element.position.width = new_size['width']
        element.position.height = new_size['height']
        
        # Snap to grid if enabled
        if canvas.snap_to_grid:
            element.position = self._snap_to_grid(element.position, canvas.grid_size)
        
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_resized(canvas_id, element)
        
        return element
    
    async def update_element_content(self, 
                                   canvas_id: str,
                                   element_id: str,
                                   content_updates: Dict[str, Any]) -> ReportElement:
        """
        Update element content
        
        Args:
            canvas_id: Canvas identifier
            element_id: Element identifier
            content_updates: Content updates
        
        Returns:
            Updated element
        """
        canvas = await self.get_canvas(canvas_id)
        element = self._find_element(canvas, element_id)
        
        if not element:
            raise ValueError(f"Element {element_id} not found")
        
        # Update content
        element.content.update(content_updates)
        canvas.updated_at = datetime.utcnow()
        
        # Refresh element data if connected to data source
        if element.data_connection:
            await self._refresh_element_data(element)
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_updated(canvas_id, element)
        
        return element
    
    async def update_element_style(self, 
                                  canvas_id: str,
                                  element_id: str,
                                  style_updates: Dict[str, Any]) -> ReportElement:
        """
        Update element styling
        
        Args:
            canvas_id: Canvas identifier
            element_id: Element identifier
            style_updates: Style updates
        
        Returns:
            Updated element
        """
        canvas = await self.get_canvas(canvas_id)
        element = self._find_element(canvas, element_id)
        
        if not element:
            raise ValueError(f"Element {element_id} not found")
        
        # Update style
        for key, value in style_updates.items():
            if hasattr(element.style, key):
                setattr(element.style, key, value)
        
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_styled(canvas_id, element)
        
        return element
    
    async def delete_element(self, canvas_id: str, element_id: str):
        """
        Delete element from canvas
        
        Args:
            canvas_id: Canvas identifier
            element_id: Element identifier
        """
        canvas = await self.get_canvas(canvas_id)
        
        # Find and remove element
        element_to_remove = None
        for i, element in enumerate(canvas.elements):
            if element.id == element_id:
                element_to_remove = element
                canvas.elements.pop(i)
                break
        
        if not element_to_remove:
            raise ValueError(f"Element {element_id} not found")
        
        # Remove children if it's a container
        if element_to_remove.type == ElementType.CONTAINER:
            for child_id in element_to_remove.children:
                await self.delete_element(canvas_id, child_id)
        
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_deleted(canvas_id, element_id)
    
    async def create_data_connection(self, 
                                   connection_type: DataConnectionType,
                                   config: Dict[str, Any]) -> str:
        """
        Create data connection
        
        Args:
            connection_type: Type of connection
            config: Connection configuration
        
        Returns:
            Connection ID
        """
        connection_id = str(uuid.uuid4())
        
        connection = DataConnection(
            id=connection_id,
            type=connection_type,
            config=config
        )
        
        self.data_connections[connection_id] = connection
        
        # Test connection
        await self._test_data_connection(connection)
        
        # Cache connection
        await self._cache_data_connection(connection_id, connection)
        
        return connection_id
    
    async def connect_element_to_data(self, 
                                    canvas_id: str,
                                    element_id: str,
                                    data_connection_id: str) -> ReportElement:
        """
        Connect element to data source
        
        Args:
            canvas_id: Canvas identifier
            element_id: Element identifier
            data_connection_id: Data connection identifier
        
        Returns:
            Updated element
        """
        canvas = await self.get_canvas(canvas_id)
        element = self._find_element(canvas, element_id)
        
        if not element:
            raise ValueError(f"Element {element_id} not found")
        
        data_connection = self.data_connections.get(data_connection_id)
        if not data_connection:
            raise ValueError(f"Data connection {data_connection_id} not found")
        
        # Connect element to data
        element.data_connection = data_connection
        
        # Refresh element with new data
        await self._refresh_element_data(element)
        
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_element_data_connected(canvas_id, element)
        
        return element
    
    async def create_element_from_template(self, 
                                         canvas_id: str,
                                         template_id: str,
                                         position: Position,
                                         customizations: Optional[Dict[str, Any]] = None) -> str:
        """
        Create element from template
        
        Args:
            canvas_id: Canvas identifier
            template_id: Template identifier
            position: Element position
            customizations: Optional customizations
        
        Returns:
            Created element ID
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Merge template config with customizations
        content = template.default_config.copy()
        if customizations:
            content.update(customizations)
        
        # Create element
        return await self.add_element(
            canvas_id=canvas_id,
            element_type=template.type,
            position=position,
            content=content
        )
    
    async def group_elements(self, 
                           canvas_id: str,
                           element_ids: List[str],
                           container_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Group elements into container
        
        Args:
            canvas_id: Canvas identifier
            element_ids: List of element IDs to group
            container_config: Container configuration
        
        Returns:
            Container element ID
        """
        canvas = await self.get_canvas(canvas_id)
        
        # Find elements to group
        elements_to_group = []
        for element_id in element_ids:
            element = self._find_element(canvas, element_id)
            if element:
                elements_to_group.append(element)
        
        if not elements_to_group:
            raise ValueError("No valid elements found to group")
        
        # Calculate container bounds
        min_x = min(el.position.x for el in elements_to_group)
        min_y = min(el.position.y for el in elements_to_group)
        max_x = max(el.position.x + el.position.width for el in elements_to_group)
        max_y = max(el.position.y + el.position.height for el in elements_to_group)
        
        # Create container
        container_position = Position(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y
        )
        
        container_content = container_config or {"layout": "absolute"}
        
        container_id = await self.add_element(
            canvas_id=canvas_id,
            element_type=ElementType.CONTAINER,
            position=container_position,
            content=container_content
        )
        
        # Update child elements
        container = self._find_element(canvas, container_id)
        for element in elements_to_group:
            element.parent_id = container_id
            container.children.append(element.id)
            
            # Adjust child positions relative to container
            element.position.x -= min_x
            element.position.y -= min_y
        
        canvas.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_canvas(canvas_id, canvas)
        
        # Notify connected clients
        await self._notify_elements_grouped(canvas_id, container_id, element_ids)
        
        return container_id
    
    async def preview_report(self, canvas_id: str) -> Dict[str, Any]:
        """
        Generate report preview
        
        Args:
            canvas_id: Canvas identifier
        
        Returns:
            Preview data
        """
        canvas = await self.get_canvas(canvas_id)
        
        preview_data = {
            'canvas': asdict(canvas),
            'elements': [],
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Generate preview for each element
        for element in canvas.elements:
            element_preview = await self._generate_element_preview(element)
            preview_data['elements'].append(element_preview)
        
        return preview_data
    
    async def export_report(self, 
                           canvas_id: str,
                           export_format: str = "pdf") -> str:
        """
        Export report to specified format
        
        Args:
            canvas_id: Canvas identifier
            export_format: Export format
        
        Returns:
            Export file path
        """
        canvas = await self.get_canvas(canvas_id)
        
        # Convert canvas to report template
        report_template = await self._canvas_to_report_template(canvas)
        
        # Generate report
        report_instance = await self.report_builder.generate_report(
            template_id=report_template.id,
            parameters={},
            output_formats=[export_format],
            generated_by="drag_drop_builder"
        )
        
        return report_instance.file_paths.get(export_format)
    
    async def get_canvas(self, canvas_id: str) -> Optional[ReportCanvas]:
        """Get canvas by ID"""
        if canvas_id in self.canvases:
            return self.canvases[canvas_id]
        
        # Try to load from cache
        cached_canvas = await self._load_canvas_from_cache(canvas_id)
        if cached_canvas:
            self.canvases[canvas_id] = cached_canvas
        
        return cached_canvas
    
    def get_available_templates(self) -> List[ElementTemplate]:
        """Get all available element templates"""
        return list(self.templates.values())
    
    def get_templates_by_category(self, category: str) -> List[ElementTemplate]:
        """Get templates by category"""
        return [
            template for template in self.templates.values()
            if template.category == category
        ]
    
    async def connect_to_session(self, 
                                canvas_id: str,
                                websocket: WebSocket,
                                user_id: str):
        """Connect to collaborative editing session"""
        if canvas_id not in self.active_sessions:
            self.active_sessions[canvas_id] = []
        
        self.active_sessions[canvas_id].append(websocket)
        
        # Send current canvas state
        canvas = await self.get_canvas(canvas_id)
        if canvas:
            await websocket.send_json({
                'type': 'canvas_state',
                'canvas': asdict(canvas)
            })
        
        # Notify other users
        await self._notify_user_joined_session(canvas_id, user_id, websocket)
    
    def _initialize_default_templates(self):
        """Initialize default element templates"""
        templates = [
            ElementTemplate(
                id="text_header",
                name="Header Text",
                type=ElementType.TEXT,
                preview_image="/templates/text_header.png",
                default_config={
                    "text": "Header Text",
                    "font_size": 24,
                    "font_weight": "bold",
                    "color": "#333333"
                },
                description="Large header text",
                category="Text",
                tags=["header", "title"]
            ),
            ElementTemplate(
                id="bar_chart",
                name="Bar Chart",
                type=ElementType.CHART,
                preview_image="/templates/bar_chart.png",
                default_config={
                    "chart_type": "bar",
                    "title": "Bar Chart",
                    "x_axis": "Category",
                    "y_axis": "Value"
                },
                description="Simple bar chart",
                category="Charts",
                tags=["chart", "bar", "visualization"]
            ),
            ElementTemplate(
                id="kpi_metric",
                name="KPI Metric",
                type=ElementType.KPI,
                preview_image="/templates/kpi_metric.png",
                default_config={
                    "metric_name": "KPI",
                    "value": 0,
                    "format": "number",
                    "trend": "up",
                    "comparison_value": None
                },
                description="Key performance indicator",
                category="Metrics",
                tags=["kpi", "metric", "performance"]
            ),
            ElementTemplate(
                id="data_table",
                name="Data Table",
                type=ElementType.TABLE,
                preview_image="/templates/data_table.png",
                default_config={
                    "columns": [],
                    "sortable": True,
                    "paginated": True,
                    "page_size": 10
                },
                description="Data table with sorting and pagination",
                category="Tables",
                tags=["table", "data", "grid"]
            )
        ]
        
        for template in templates:
            self.templates[template.id] = template
    
    def _find_element(self, canvas: ReportCanvas, element_id: str) -> Optional[ReportElement]:
        """Find element in canvas by ID"""
        for element in canvas.elements:
            if element.id == element_id:
                return element
        return None
    
    def _snap_to_grid(self, position: Position, grid_size: int) -> Position:
        """Snap position to grid"""
        snapped_position = Position(
            x=round(position.x / grid_size) * grid_size,
            y=round(position.y / grid_size) * grid_size,
            width=round(position.width / grid_size) * grid_size,
            height=round(position.height / grid_size) * grid_size,
            z_index=position.z_index
        )
        return snapped_position
    
    async def _check_collision(self, 
                              canvas: ReportCanvas,
                              element_id: str,
                              position: Position) -> bool:
        """Check if position collides with other elements"""
        for element in canvas.elements:
            if element.id == element_id:
                continue
            
            # Check overlap
            if (position.x < element.position.x + element.position.width and
                position.x + position.width > element.position.x and
                position.y < element.position.y + element.position.height and
                position.y + position.height > element.position.y):
                return True
        
        return False
    
    async def _resolve_collision(self, 
                                canvas: ReportCanvas,
                                element_id: str,
                                position: Position) -> Position:
        """Resolve collision by finding nearby free space"""
        # Simple implementation - move to nearest free space
        offset = 10
        attempts = 0
        max_attempts = 100
        
        while await self._check_collision(canvas, element_id, position) and attempts < max_attempts:
            position.x += offset
            position.y += offset
            attempts += 1
        
        return position
    
    async def _refresh_element_data(self, element: ReportElement):
        """Refresh element data from its data connection"""
        if not element.data_connection:
            return
        
        try:
            if element.data_connection.type == DataConnectionType.SQL_QUERY:
                data = await self._execute_sql_query(element.data_connection.config['query'])
                element.content['data'] = data.to_dict('records')
            elif element.data_connection.type == DataConnectionType.API_ENDPOINT:
                data = await self._fetch_api_data(element.data_connection.config['url'])
                element.content['data'] = data
            
            element.data_connection.last_updated = datetime.utcnow()
            
        except Exception as e:
            print(f"Error refreshing element data: {str(e)}")
    
    async def _generate_element_preview(self, element: ReportElement) -> Dict[str, Any]:
        """Generate preview for element"""
        preview = {
            'id': element.id,
            'type': element.type.value,
            'position': asdict(element.position),
            'style': asdict(element.style),
            'content': element.content.copy()
        }
        
        # Generate specific preview based on element type
        if element.type == ElementType.CHART:
            # Generate chart preview
            preview['chart_preview'] = await self._generate_chart_preview(element)
        elif element.type == ElementType.TABLE:
            # Generate table preview
            preview['table_preview'] = await self._generate_table_preview(element)
        elif element.type == ElementType.KPI:
            # Generate KPI preview
            preview['kpi_preview'] = await self._generate_kpi_preview(element)
        
        return preview
    
    async def _generate_chart_preview(self, element: ReportElement) -> Dict[str, Any]:
        """Generate chart preview"""
        # Mock chart data for preview
        return {
            'chart_type': element.content.get('chart_type', 'bar'),
            'data_points': 10,
            'has_data': bool(element.content.get('data'))
        }
    
    async def _generate_table_preview(self, element: ReportElement) -> Dict[str, Any]:
        """Generate table preview"""
        data = element.content.get('data', [])
        return {
            'row_count': len(data),
            'column_count': len(data[0].keys()) if data else 0,
            'has_data': bool(data)
        }
    
    async def _generate_kpi_preview(self, element: ReportElement) -> Dict[str, Any]:
        """Generate KPI preview"""
        return {
            'metric_name': element.content.get('metric_name', 'KPI'),
            'value': element.content.get('value', 0),
            'formatted_value': self._format_kpi_value(element.content.get('value', 0))
        }
    
    def _format_kpi_value(self, value: Union[int, float]) -> str:
        """Format KPI value for display"""
        if isinstance(value, (int, float)):
            if value >= 1000000:
                return f"{value / 1000000:.1f}M"
            elif value >= 1000:
                return f"{value / 1000:.1f}K"
            else:
                return str(value)
        return str(value)
    
    async def _test_data_connection(self, connection: DataConnection):
        """Test data connection"""
        try:
            if connection.type == DataConnectionType.SQL_QUERY:
                # Test SQL connection
                test_query = "SELECT 1"
                await self._execute_sql_query(test_query)
            elif connection.type == DataConnectionType.API_ENDPOINT:
                # Test API endpoint
                await self._fetch_api_data(connection.config['url'])
            
            return True
        except Exception as e:
            print(f"Data connection test failed: {str(e)}")
            return False
    
    async def _execute_sql_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        # This would integrate with your database
        # For now, return mock data
        return pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
    
    async def _fetch_api_data(self, url: str) -> Dict[str, Any]:
        """Fetch data from API endpoint"""
        # This would make actual API call
        # For now, return mock data
        return {
            'data': [
                {'name': 'Item 1', 'value': 100},
                {'name': 'Item 2', 'value': 200}
            ]
        }
    
    async def _cache_canvas(self, canvas_id: str, canvas: ReportCanvas):
        """Cache canvas in Redis"""
        await self.redis_client.set(
            f"canvas:{canvas_id}",
            json.dumps(asdict(canvas), default=str),
            ex=3600  # 1 hour expiry
        )
    
    async def _cache_data_connection(self, connection_id: str, connection: DataConnection):
        """Cache data connection in Redis"""
        await self.redis_client.set(
            f"data_connection:{connection_id}",
            json.dumps(asdict(connection), default=str),
            ex=3600  # 1 hour expiry
        )
    
    async def _load_canvas_from_cache(self, canvas_id: str) -> Optional[ReportCanvas]:
        """Load canvas from cache"""
        cached_data = await self.redis_client.get(f"canvas:{canvas_id}")
        if cached_data:
            data = json.loads(cached_data)
            return ReportCanvas(**data)
        return None
    
    async def _notify_element_added(self, canvas_id: str, element: ReportElement):
        """Notify clients of new element"""
        await self._notify_clients(canvas_id, {
            'type': 'element_added',
            'element': asdict(element)
        })
    
    async def _notify_element_moved(self, canvas_id: str, element: ReportElement):
        """Notify clients of element movement"""
        await self._notify_clients(canvas_id, {
            'type': 'element_moved',
            'element_id': element.id,
            'position': asdict(element.position)
        })
    
    async def _notify_clients(self, canvas_id: str, message: Dict[str, Any]):
        """Notify all connected clients"""
        if canvas_id in self.active_sessions:
            for websocket in self.active_sessions[canvas_id]:
                try:
                    await websocket.send_json(message)
                except:
                    pass