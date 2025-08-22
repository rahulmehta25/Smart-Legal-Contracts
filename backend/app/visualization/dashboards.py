"""
Dynamic Dashboard Management System

Provides comprehensive dashboard creation and management capabilities with real-time updates,
collaborative features, and advanced analytics integration.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from fastapi import WebSocket
import redis
from sqlalchemy.orm import Session

from .charts import ChartEngine


class WidgetType(Enum):
    """Types of dashboard widgets"""
    CHART = "chart"
    KPI = "kpi"
    TABLE = "table"
    TEXT = "text"
    MAP = "map"
    FILTER = "filter"
    IFRAME = "iframe"
    CUSTOM = "custom"


class RefreshMode(Enum):
    """Dashboard refresh modes"""
    MANUAL = "manual"
    AUTO = "auto"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"


@dataclass
class Widget:
    """Dashboard widget configuration"""
    id: str
    type: WidgetType
    title: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    data_source: str
    refresh_interval: Optional[int] = None
    filters: List[Dict] = None
    permissions: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.filters is None:
            self.filters = []
        if self.permissions is None:
            self.permissions = []


@dataclass
class Dashboard:
    """Dashboard configuration"""
    id: str
    name: str
    description: str
    widgets: List[Widget]
    layout: Dict[str, Any]
    theme: str = "light"
    refresh_mode: RefreshMode = RefreshMode.MANUAL
    refresh_interval: int = 300  # seconds
    filters: List[Dict] = None
    permissions: Dict[str, List[str]] = None
    tags: List[str] = None
    owner_id: str = None
    shared_with: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.filters is None:
            self.filters = []
        if self.permissions is None:
            self.permissions = {}
        if self.tags is None:
            self.tags = []
        if self.shared_with is None:
            self.shared_with = []


class DashboardManager:
    """
    Advanced dashboard management system with real-time capabilities
    """
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis_client = redis_client
        self.db_session = db_session
        self.chart_engine = ChartEngine()
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.dashboard_cache = {}
        
    async def create_dashboard(self, 
                             name: str,
                             description: str,
                             owner_id: str,
                             initial_widgets: List[Dict] = None) -> Dashboard:
        """
        Create new dashboard with optional initial widgets
        
        Args:
            name: Dashboard name
            description: Dashboard description
            owner_id: ID of dashboard owner
            initial_widgets: Optional list of initial widget configurations
        
        Returns:
            Created dashboard object
        """
        dashboard_id = str(uuid.uuid4())
        
        widgets = []
        if initial_widgets:
            for widget_config in initial_widgets:
                widget = Widget(
                    id=str(uuid.uuid4()),
                    type=WidgetType(widget_config['type']),
                    title=widget_config['title'],
                    position=widget_config['position'],
                    config=widget_config.get('config', {}),
                    data_source=widget_config['data_source']
                )
                widgets.append(widget)
        
        dashboard = Dashboard(
            id=dashboard_id,
            name=name,
            description=description,
            widgets=widgets,
            layout={"grid": {"columns": 12, "rows": "auto"}},
            owner_id=owner_id
        )
        
        # Cache dashboard
        await self._cache_dashboard(dashboard)
        
        # Store in database
        await self._persist_dashboard(dashboard)
        
        return dashboard
    
    async def add_widget(self, 
                        dashboard_id: str,
                        widget_config: Dict[str, Any]) -> Widget:
        """
        Add widget to existing dashboard
        
        Args:
            dashboard_id: ID of dashboard
            widget_config: Widget configuration
        
        Returns:
            Created widget
        """
        dashboard = await self.get_dashboard(dashboard_id)
        
        widget = Widget(
            id=str(uuid.uuid4()),
            type=WidgetType(widget_config['type']),
            title=widget_config['title'],
            position=widget_config['position'],
            config=widget_config.get('config', {}),
            data_source=widget_config['data_source'],
            refresh_interval=widget_config.get('refresh_interval')
        )
        
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.utcnow()
        
        # Update cache and database
        await self._cache_dashboard(dashboard)
        await self._persist_dashboard(dashboard)
        
        # Notify connected clients
        await self._notify_dashboard_update(dashboard_id, {
            'type': 'widget_added',
            'widget': asdict(widget)
        })
        
        return widget
    
    async def update_widget(self, 
                           dashboard_id: str,
                           widget_id: str,
                           updates: Dict[str, Any]) -> Widget:
        """
        Update existing widget
        
        Args:
            dashboard_id: ID of dashboard
            widget_id: ID of widget to update
            updates: Dictionary of updates to apply
        
        Returns:
            Updated widget
        """
        dashboard = await self.get_dashboard(dashboard_id)
        
        widget = next((w for w in dashboard.widgets if w.id == widget_id), None)
        if not widget:
            raise ValueError(f"Widget {widget_id} not found")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(widget, key):
                setattr(widget, key, value)
        
        widget.updated_at = datetime.utcnow()
        dashboard.updated_at = datetime.utcnow()
        
        # Update cache and database
        await self._cache_dashboard(dashboard)
        await self._persist_dashboard(dashboard)
        
        # Notify connected clients
        await self._notify_dashboard_update(dashboard_id, {
            'type': 'widget_updated',
            'widget_id': widget_id,
            'updates': updates
        })
        
        return widget
    
    async def remove_widget(self, dashboard_id: str, widget_id: str):
        """
        Remove widget from dashboard
        
        Args:
            dashboard_id: ID of dashboard
            widget_id: ID of widget to remove
        """
        dashboard = await self.get_dashboard(dashboard_id)
        
        dashboard.widgets = [w for w in dashboard.widgets if w.id != widget_id]
        dashboard.updated_at = datetime.utcnow()
        
        # Update cache and database
        await self._cache_dashboard(dashboard)
        await self._persist_dashboard(dashboard)
        
        # Notify connected clients
        await self._notify_dashboard_update(dashboard_id, {
            'type': 'widget_removed',
            'widget_id': widget_id
        })
    
    async def get_dashboard(self, dashboard_id: str) -> Dashboard:
        """
        Retrieve dashboard by ID
        
        Args:
            dashboard_id: ID of dashboard
        
        Returns:
            Dashboard object
        """
        # Try cache first
        cached = await self._get_cached_dashboard(dashboard_id)
        if cached:
            return cached
        
        # Load from database
        dashboard = await self._load_dashboard_from_db(dashboard_id)
        if dashboard:
            await self._cache_dashboard(dashboard)
        
        return dashboard
    
    async def get_widget_data(self, 
                             dashboard_id: str,
                             widget_id: str,
                             filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get data for specific widget
        
        Args:
            dashboard_id: ID of dashboard
            widget_id: ID of widget
            filters: Optional filters to apply
        
        Returns:
            Widget data dictionary
        """
        dashboard = await self.get_dashboard(dashboard_id)
        widget = next((w for w in dashboard.widgets if w.id == widget_id), None)
        
        if not widget:
            raise ValueError(f"Widget {widget_id} not found")
        
        # Apply dashboard-level filters
        combined_filters = {**(filters or {}), **dict(dashboard.filters or [])}
        
        # Get data based on widget type and data source
        data = await self._fetch_widget_data(widget, combined_filters)
        
        return {
            'widget_id': widget_id,
            'data': data,
            'last_updated': datetime.utcnow().isoformat(),
            'filters_applied': combined_filters
        }
    
    async def create_kpi_widget(self, 
                               dashboard_id: str,
                               title: str,
                               metric_query: str,
                               position: Dict[str, int],
                               target_value: Optional[float] = None,
                               format_spec: str = ",.0f") -> Widget:
        """
        Create KPI widget with advanced formatting
        
        Args:
            dashboard_id: ID of dashboard
            title: Widget title
            metric_query: Query to fetch KPI value
            position: Widget position
            target_value: Optional target value for comparison
            format_spec: Number formatting specification
        
        Returns:
            Created KPI widget
        """
        widget_config = {
            'type': 'kpi',
            'title': title,
            'position': position,
            'data_source': 'sql_query',
            'config': {
                'query': metric_query,
                'target_value': target_value,
                'format_spec': format_spec,
                'trend_calculation': True,
                'comparison_period': '7d'
            }
        }
        
        return await self.add_widget(dashboard_id, widget_config)
    
    async def create_filter_widget(self, 
                                  dashboard_id: str,
                                  title: str,
                                  filter_type: str,
                                  options_query: str,
                                  position: Dict[str, int]) -> Widget:
        """
        Create filter widget for dashboard-wide filtering
        
        Args:
            dashboard_id: ID of dashboard
            title: Widget title
            filter_type: Type of filter (dropdown, multiselect, date_range, etc.)
            options_query: Query to fetch filter options
            position: Widget position
        
        Returns:
            Created filter widget
        """
        widget_config = {
            'type': 'filter',
            'title': title,
            'position': position,
            'data_source': 'sql_query',
            'config': {
                'filter_type': filter_type,
                'options_query': options_query,
                'default_value': None,
                'affects_widgets': 'all'
            }
        }
        
        return await self.add_widget(dashboard_id, widget_config)
    
    async def apply_dashboard_filter(self, 
                                   dashboard_id: str,
                                   filter_name: str,
                                   filter_value: Any):
        """
        Apply filter to entire dashboard
        
        Args:
            dashboard_id: ID of dashboard
            filter_name: Name of filter to apply
            filter_value: Value to filter by
        """
        dashboard = await self.get_dashboard(dashboard_id)
        
        # Update dashboard filters
        existing_filters = {f['name']: f['value'] for f in dashboard.filters}
        existing_filters[filter_name] = filter_value
        
        dashboard.filters = [{'name': k, 'value': v} for k, v in existing_filters.items()]
        dashboard.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_dashboard(dashboard)
        
        # Refresh all affected widgets
        affected_widgets = []
        for widget in dashboard.widgets:
            if widget.config.get('filter_affected', True):
                widget_data = await self.get_widget_data(dashboard_id, widget.id, existing_filters)
                affected_widgets.append(widget_data)
        
        # Notify connected clients
        await self._notify_dashboard_update(dashboard_id, {
            'type': 'filter_applied',
            'filter_name': filter_name,
            'filter_value': filter_value,
            'affected_widgets': affected_widgets
        })
    
    async def clone_dashboard(self, 
                             dashboard_id: str,
                             new_name: str,
                             owner_id: str) -> Dashboard:
        """
        Clone existing dashboard
        
        Args:
            dashboard_id: ID of dashboard to clone
            new_name: Name for cloned dashboard
            owner_id: Owner of cloned dashboard
        
        Returns:
            Cloned dashboard
        """
        original = await self.get_dashboard(dashboard_id)
        
        # Clone widgets
        cloned_widgets = []
        for widget in original.widgets:
            cloned_widget = Widget(
                id=str(uuid.uuid4()),
                type=widget.type,
                title=widget.title,
                position=widget.position.copy(),
                config=widget.config.copy(),
                data_source=widget.data_source,
                refresh_interval=widget.refresh_interval,
                filters=widget.filters.copy() if widget.filters else []
            )
            cloned_widgets.append(cloned_widget)
        
        # Create cloned dashboard
        cloned_dashboard = Dashboard(
            id=str(uuid.uuid4()),
            name=new_name,
            description=f"Cloned from: {original.description}",
            widgets=cloned_widgets,
            layout=original.layout.copy(),
            theme=original.theme,
            refresh_mode=original.refresh_mode,
            refresh_interval=original.refresh_interval,
            filters=original.filters.copy() if original.filters else [],
            tags=original.tags.copy() if original.tags else [],
            owner_id=owner_id
        )
        
        # Cache and persist
        await self._cache_dashboard(cloned_dashboard)
        await self._persist_dashboard(cloned_dashboard)
        
        return cloned_dashboard
    
    async def get_dashboard_performance_metrics(self, dashboard_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for dashboard
        
        Args:
            dashboard_id: ID of dashboard
        
        Returns:
            Performance metrics dictionary
        """
        dashboard = await self.get_dashboard(dashboard_id)
        
        # Calculate metrics
        total_widgets = len(dashboard.widgets)
        
        # Get load times from cache
        load_times = []
        for widget in dashboard.widgets:
            cache_key = f"widget_load_time:{widget.id}"
            load_time = await self.redis_client.get(cache_key)
            if load_time:
                load_times.append(float(load_time))
        
        avg_load_time = sum(load_times) / len(load_times) if load_times else 0
        
        # Get usage statistics
        views_key = f"dashboard_views:{dashboard_id}"
        total_views = await self.redis_client.get(views_key) or 0
        
        return {
            'dashboard_id': dashboard_id,
            'total_widgets': total_widgets,
            'average_load_time': avg_load_time,
            'total_views': int(total_views),
            'last_accessed': dashboard.updated_at.isoformat(),
            'cache_hit_rate': await self._calculate_cache_hit_rate(dashboard_id),
            'data_freshness': await self._calculate_data_freshness(dashboard_id)
        }
    
    async def export_dashboard_config(self, dashboard_id: str) -> Dict[str, Any]:
        """
        Export dashboard configuration for backup or migration
        
        Args:
            dashboard_id: ID of dashboard to export
        
        Returns:
            Exportable dashboard configuration
        """
        dashboard = await self.get_dashboard(dashboard_id)
        
        config = {
            'dashboard': asdict(dashboard),
            'export_timestamp': datetime.utcnow().isoformat(),
            'version': '1.0',
            'metadata': {
                'total_widgets': len(dashboard.widgets),
                'widget_types': [w.type.value for w in dashboard.widgets]
            }
        }
        
        return config
    
    async def import_dashboard_config(self, 
                                    config: Dict[str, Any],
                                    owner_id: str,
                                    new_name: Optional[str] = None) -> Dashboard:
        """
        Import dashboard from configuration
        
        Args:
            config: Dashboard configuration to import
            owner_id: Owner for imported dashboard
            new_name: Optional new name for dashboard
        
        Returns:
            Imported dashboard
        """
        dashboard_data = config['dashboard']
        
        # Create new dashboard ID
        new_dashboard_id = str(uuid.uuid4())
        
        # Recreate widgets with new IDs
        widgets = []
        for widget_data in dashboard_data['widgets']:
            widget = Widget(
                id=str(uuid.uuid4()),
                type=WidgetType(widget_data['type']),
                title=widget_data['title'],
                position=widget_data['position'],
                config=widget_data['config'],
                data_source=widget_data['data_source'],
                refresh_interval=widget_data.get('refresh_interval'),
                filters=widget_data.get('filters', [])
            )
            widgets.append(widget)
        
        # Create dashboard
        dashboard = Dashboard(
            id=new_dashboard_id,
            name=new_name or f"{dashboard_data['name']} (Imported)",
            description=dashboard_data['description'],
            widgets=widgets,
            layout=dashboard_data['layout'],
            theme=dashboard_data.get('theme', 'light'),
            refresh_mode=RefreshMode(dashboard_data.get('refresh_mode', 'manual')),
            refresh_interval=dashboard_data.get('refresh_interval', 300),
            filters=dashboard_data.get('filters', []),
            tags=dashboard_data.get('tags', []),
            owner_id=owner_id
        )
        
        # Cache and persist
        await self._cache_dashboard(dashboard)
        await self._persist_dashboard(dashboard)
        
        return dashboard
    
    async def connect_websocket(self, dashboard_id: str, websocket: WebSocket):
        """
        Connect WebSocket for real-time dashboard updates
        
        Args:
            dashboard_id: ID of dashboard
            websocket: WebSocket connection
        """
        if dashboard_id not in self.active_connections:
            self.active_connections[dashboard_id] = []
        
        self.active_connections[dashboard_id].append(websocket)
        
        # Send initial dashboard state
        dashboard = await self.get_dashboard(dashboard_id)
        await websocket.send_json({
            'type': 'dashboard_state',
            'dashboard': asdict(dashboard)
        })
    
    async def disconnect_websocket(self, dashboard_id: str, websocket: WebSocket):
        """
        Disconnect WebSocket
        
        Args:
            dashboard_id: ID of dashboard
            websocket: WebSocket connection to remove
        """
        if dashboard_id in self.active_connections:
            self.active_connections[dashboard_id].remove(websocket)
            if not self.active_connections[dashboard_id]:
                del self.active_connections[dashboard_id]
    
    async def _cache_dashboard(self, dashboard: Dashboard):
        """Cache dashboard in Redis"""
        cache_key = f"dashboard:{dashboard.id}"
        await self.redis_client.set(
            cache_key,
            json.dumps(asdict(dashboard), default=str),
            ex=3600  # 1 hour expiry
        )
    
    async def _get_cached_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard from cache"""
        cache_key = f"dashboard:{dashboard_id}"
        cached_data = await self.redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            # Reconstruct dashboard object
            widgets = [Widget(**w) for w in data['widgets']]
            data['widgets'] = widgets
            return Dashboard(**data)
        
        return None
    
    async def _persist_dashboard(self, dashboard: Dashboard):
        """Persist dashboard to database"""
        # This would integrate with your database layer
        # Implementation depends on your ORM/database setup
        pass
    
    async def _load_dashboard_from_db(self, dashboard_id: str) -> Optional[Dashboard]:
        """Load dashboard from database"""
        # This would integrate with your database layer
        # Implementation depends on your ORM/database setup
        pass
    
    async def _fetch_widget_data(self, widget: Widget, filters: Dict[str, Any]) -> Any:
        """Fetch data for widget based on its configuration"""
        # This would integrate with your data sources
        # Implementation depends on your data layer
        pass
    
    async def _notify_dashboard_update(self, dashboard_id: str, update: Dict[str, Any]):
        """Notify connected WebSocket clients of dashboard updates"""
        if dashboard_id in self.active_connections:
            message = json.dumps(update, default=str)
            for websocket in self.active_connections[dashboard_id]:
                try:
                    await websocket.send_text(message)
                except:
                    # Handle disconnected websockets
                    pass
    
    async def _calculate_cache_hit_rate(self, dashboard_id: str) -> float:
        """Calculate cache hit rate for dashboard"""
        hits_key = f"cache_hits:{dashboard_id}"
        misses_key = f"cache_misses:{dashboard_id}"
        
        hits = int(await self.redis_client.get(hits_key) or 0)
        misses = int(await self.redis_client.get(misses_key) or 0)
        
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    async def _calculate_data_freshness(self, dashboard_id: str) -> Dict[str, Any]:
        """Calculate data freshness metrics for dashboard"""
        dashboard = await self.get_dashboard(dashboard_id)
        
        freshness_data = {}
        for widget in dashboard.widgets:
            last_update_key = f"widget_last_update:{widget.id}"
            last_update = await self.redis_client.get(last_update_key)
            
            if last_update:
                last_update_time = datetime.fromisoformat(last_update.decode())
                age_minutes = (datetime.utcnow() - last_update_time).total_seconds() / 60
                freshness_data[widget.id] = {
                    'last_update': last_update_time.isoformat(),
                    'age_minutes': age_minutes,
                    'is_stale': age_minutes > (widget.refresh_interval or 300) / 60
                }
        
        return freshness_data