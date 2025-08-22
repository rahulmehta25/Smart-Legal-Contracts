"""
Interactive Visualization Features

Provides advanced interactive capabilities including drill-down, cross-filtering,
collaborative annotations, what-if scenarios, and AR/VR data exploration.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from fastapi import WebSocket
import redis

from .charts import ChartEngine


class InteractionType(Enum):
    """Types of chart interactions"""
    CLICK = "click"
    HOVER = "hover"
    SELECT = "select"
    BRUSH = "brush"
    ZOOM = "zoom"
    PAN = "pan"
    DRILL_DOWN = "drill_down"
    DRILL_UP = "drill_up"
    CROSS_FILTER = "cross_filter"


class AnnotationType(Enum):
    """Types of annotations"""
    TEXT = "text"
    ARROW = "arrow"
    SHAPE = "shape"
    HIGHLIGHT = "highlight"
    COMMENT = "comment"
    STICKY_NOTE = "sticky_note"


class ScenarioType(Enum):
    """Types of what-if scenarios"""
    PARAMETER_CHANGE = "parameter_change"
    DATA_MODIFICATION = "data_modification"
    FORECASTING = "forecasting"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"


@dataclass
class Interaction:
    """Chart interaction definition"""
    id: str
    type: InteractionType
    source_element: str
    target_chart: Optional[str] = None
    action: Dict[str, Any] = None
    conditions: List[Dict] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.action is None:
            self.action = {}
        if self.conditions is None:
            self.conditions = []


@dataclass
class Annotation:
    """Chart annotation definition"""
    id: str
    type: AnnotationType
    chart_id: str
    position: Dict[str, float]
    content: str
    author: str
    created_at: datetime = None
    updated_at: datetime = None
    style: Dict[str, Any] = None
    visible: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.style is None:
            self.style = {}


@dataclass
class Scenario:
    """What-if scenario definition"""
    id: str
    name: str
    type: ScenarioType
    base_chart_id: str
    parameters: Dict[str, Any]
    created_by: str
    created_at: datetime = None
    results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class DrillPath:
    """Drill-down/up path definition"""
    levels: List[str]
    current_level: int = 0
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class InteractiveManager:
    """
    Advanced interactive visualization manager
    """
    
    def __init__(self, chart_engine: ChartEngine, redis_client: redis.Redis):
        self.chart_engine = chart_engine
        self.redis_client = redis_client
        
        # Storage for interactive elements
        self.interactions: Dict[str, Interaction] = {}
        self.annotations: Dict[str, Annotation] = {}
        self.scenarios: Dict[str, Scenario] = {}
        self.drill_paths: Dict[str, DrillPath] = {}
        
        # Active connections for real-time collaboration
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        
        # Cross-filtering state
        self.cross_filter_state: Dict[str, Dict[str, Any]] = {}
    
    async def add_drill_down_capability(self, 
                                       chart_id: str,
                                       drill_levels: List[str],
                                       data_source_config: Dict[str, Any]) -> str:
        """
        Add drill-down capability to chart
        
        Args:
            chart_id: Chart identifier
            drill_levels: List of drill levels (e.g., ['country', 'state', 'city'])
            data_source_config: Configuration for data retrieval at each level
        
        Returns:
            Drill path ID
        """
        drill_path_id = str(uuid.uuid4())
        
        drill_path = DrillPath(levels=drill_levels)
        self.drill_paths[drill_path_id] = drill_path
        
        # Create interaction for drill-down
        interaction = Interaction(
            id=str(uuid.uuid4()),
            type=InteractionType.DRILL_DOWN,
            source_element=chart_id,
            action={
                'drill_path_id': drill_path_id,
                'data_source_config': data_source_config
            }
        )
        
        self.interactions[interaction.id] = interaction
        
        # Cache drill path
        await self._cache_drill_path(drill_path_id, drill_path)
        
        return drill_path_id
    
    async def execute_drill_down(self, 
                                drill_path_id: str,
                                selected_value: Any,
                                chart_config: Dict[str, Any]) -> go.Figure:
        """
        Execute drill-down operation
        
        Args:
            drill_path_id: Drill path identifier
            selected_value: Selected value to drill down on
            chart_config: Chart configuration
        
        Returns:
            Updated chart with drilled-down data
        """
        drill_path = self.drill_paths.get(drill_path_id)
        if not drill_path:
            raise ValueError(f"Drill path {drill_path_id} not found")
        
        # Move to next level
        if drill_path.current_level < len(drill_path.levels) - 1:
            current_level_column = drill_path.levels[drill_path.current_level]
            drill_path.filters[current_level_column] = selected_value
            drill_path.current_level += 1
            
            # Get data for next level
            next_level_data = await self._get_drill_down_data(drill_path, chart_config)
            
            # Create updated chart
            updated_chart = await self._create_drill_down_chart(next_level_data, drill_path, chart_config)
            
            # Update cache
            await self._cache_drill_path(drill_path_id, drill_path)
            
            return updated_chart
        else:
            raise ValueError("Already at deepest drill level")
    
    async def execute_drill_up(self, 
                              drill_path_id: str,
                              chart_config: Dict[str, Any]) -> go.Figure:
        """
        Execute drill-up operation
        
        Args:
            drill_path_id: Drill path identifier
            chart_config: Chart configuration
        
        Returns:
            Updated chart with drilled-up data
        """
        drill_path = self.drill_paths.get(drill_path_id)
        if not drill_path:
            raise ValueError(f"Drill path {drill_path_id} not found")
        
        # Move to previous level
        if drill_path.current_level > 0:
            drill_path.current_level -= 1
            
            # Remove filter for current level
            current_level_column = drill_path.levels[drill_path.current_level]
            drill_path.filters.pop(current_level_column, None)
            
            # Get data for previous level
            previous_level_data = await self._get_drill_down_data(drill_path, chart_config)
            
            # Create updated chart
            updated_chart = await self._create_drill_down_chart(previous_level_data, drill_path, chart_config)
            
            # Update cache
            await self._cache_drill_path(drill_path_id, drill_path)
            
            return updated_chart
        else:
            raise ValueError("Already at highest drill level")
    
    async def add_cross_filtering(self, 
                                 source_chart_id: str,
                                 target_chart_ids: List[str],
                                 filter_config: Dict[str, Any]) -> str:
        """
        Add cross-filtering capability between charts
        
        Args:
            source_chart_id: Source chart that triggers filtering
            target_chart_ids: Target charts to be filtered
            filter_config: Configuration for filtering logic
        
        Returns:
            Interaction ID
        """
        interaction_id = str(uuid.uuid4())
        
        interaction = Interaction(
            id=interaction_id,
            type=InteractionType.CROSS_FILTER,
            source_element=source_chart_id,
            action={
                'target_charts': target_chart_ids,
                'filter_config': filter_config
            }
        )
        
        self.interactions[interaction_id] = interaction
        
        # Initialize cross-filter state
        if source_chart_id not in self.cross_filter_state:
            self.cross_filter_state[source_chart_id] = {}
        
        return interaction_id
    
    async def apply_cross_filter(self, 
                                source_chart_id: str,
                                selected_data: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Apply cross-filtering to target charts
        
        Args:
            source_chart_id: Source chart ID
            selected_data: Selected data for filtering
        
        Returns:
            Dictionary of updated target charts
        """
        updated_charts = {}
        
        # Find cross-filter interactions for source chart
        cross_filter_interactions = [
            interaction for interaction in self.interactions.values()
            if (interaction.source_element == source_chart_id and 
                interaction.type == InteractionType.CROSS_FILTER)
        ]
        
        for interaction in cross_filter_interactions:
            target_charts = interaction.action.get('target_charts', [])
            filter_config = interaction.action.get('filter_config', {})
            
            # Apply filter to each target chart
            for target_chart_id in target_charts:
                filtered_chart = await self._apply_filter_to_chart(
                    target_chart_id, selected_data, filter_config
                )
                updated_charts[target_chart_id] = filtered_chart
        
        # Update cross-filter state
        self.cross_filter_state[source_chart_id] = selected_data
        
        # Notify connected clients
        await self._notify_cross_filter_update(source_chart_id, selected_data, updated_charts)
        
        return updated_charts
    
    async def add_annotation(self, 
                           chart_id: str,
                           annotation_type: AnnotationType,
                           position: Dict[str, float],
                           content: str,
                           author: str,
                           style: Optional[Dict[str, Any]] = None) -> str:
        """
        Add annotation to chart
        
        Args:
            chart_id: Chart identifier
            annotation_type: Type of annotation
            position: Position on chart (x, y coordinates)
            content: Annotation content
            author: Author of annotation
            style: Optional styling
        
        Returns:
            Annotation ID
        """
        annotation_id = str(uuid.uuid4())
        
        annotation = Annotation(
            id=annotation_id,
            type=annotation_type,
            chart_id=chart_id,
            position=position,
            content=content,
            author=author,
            style=style or {}
        )
        
        self.annotations[annotation_id] = annotation
        
        # Cache annotation
        await self._cache_annotation(annotation_id, annotation)
        
        # Notify connected clients
        await self._notify_annotation_added(chart_id, annotation)
        
        return annotation_id
    
    async def update_annotation(self, 
                               annotation_id: str,
                               updates: Dict[str, Any]) -> Annotation:
        """
        Update existing annotation
        
        Args:
            annotation_id: Annotation identifier
            updates: Updates to apply
        
        Returns:
            Updated annotation
        """
        annotation = self.annotations.get(annotation_id)
        if not annotation:
            raise ValueError(f"Annotation {annotation_id} not found")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(annotation, key):
                setattr(annotation, key, value)
        
        annotation.updated_at = datetime.utcnow()
        
        # Update cache
        await self._cache_annotation(annotation_id, annotation)
        
        # Notify connected clients
        await self._notify_annotation_updated(annotation.chart_id, annotation)
        
        return annotation
    
    async def create_what_if_scenario(self, 
                                    name: str,
                                    base_chart_id: str,
                                    scenario_type: ScenarioType,
                                    parameters: Dict[str, Any],
                                    created_by: str) -> str:
        """
        Create what-if scenario
        
        Args:
            name: Scenario name
            base_chart_id: Base chart ID
            scenario_type: Type of scenario
            parameters: Scenario parameters
            created_by: Creator user ID
        
        Returns:
            Scenario ID
        """
        scenario_id = str(uuid.uuid4())
        
        scenario = Scenario(
            id=scenario_id,
            name=name,
            type=scenario_type,
            base_chart_id=base_chart_id,
            parameters=parameters,
            created_by=created_by
        )
        
        self.scenarios[scenario_id] = scenario
        
        # Execute scenario
        results = await self._execute_scenario(scenario)
        scenario.results = results
        
        # Cache scenario
        await self._cache_scenario(scenario_id, scenario)
        
        return scenario_id
    
    async def execute_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """
        Execute what-if scenario and return results
        
        Args:
            scenario_id: Scenario identifier
        
        Returns:
            Scenario execution results
        """
        scenario = self.scenarios.get(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        return await self._execute_scenario(scenario)
    
    async def create_predictive_visualization(self, 
                                            base_data: pd.DataFrame,
                                            prediction_config: Dict[str, Any]) -> go.Figure:
        """
        Create predictive visualization with confidence intervals
        
        Args:
            base_data: Base historical data
            prediction_config: Prediction configuration
        
        Returns:
            Predictive visualization chart
        """
        # Generate predictions based on configuration
        prediction_method = prediction_config.get('method', 'linear_trend')
        prediction_periods = prediction_config.get('periods', 30)
        confidence_level = prediction_config.get('confidence_level', 0.95)
        
        if prediction_method == 'linear_trend':
            predictions = await self._generate_linear_trend_predictions(
                base_data, prediction_periods, confidence_level
            )
        elif prediction_method == 'exponential_smoothing':
            predictions = await self._generate_exponential_smoothing_predictions(
                base_data, prediction_periods, confidence_level
            )
        elif prediction_method == 'arima':
            predictions = await self._generate_arima_predictions(
                base_data, prediction_periods, confidence_level
            )
        else:
            raise ValueError(f"Unsupported prediction method: {prediction_method}")
        
        # Create visualization
        return self.chart_engine.create_predictive_visualization(
            base_data, predictions['forecast'], predictions.get('confidence_intervals')
        )
    
    async def create_ar_vr_scene(self, 
                                data_points: List[Dict[str, Any]],
                                scene_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create AR/VR data exploration scene
        
        Args:
            data_points: 3D data points
            scene_config: Scene configuration
        
        Returns:
            AR/VR scene configuration
        """
        # Process data for 3D visualization
        processed_data = await self._process_data_for_3d(data_points, scene_config)
        
        # Create A-Frame scene
        scene = self.chart_engine.create_ar_vr_data_scene(processed_data)
        
        # Add interactive elements
        if scene_config.get('enable_interaction', True):
            scene = await self._add_ar_vr_interactions(scene, scene_config)
        
        # Add data labels and tooltips
        if scene_config.get('show_labels', True):
            scene = await self._add_ar_vr_labels(scene, processed_data)
        
        return scene
    
    async def connect_collaborative_session(self, 
                                          chart_id: str,
                                          websocket: WebSocket,
                                          user_id: str):
        """
        Connect user to collaborative annotation session
        
        Args:
            chart_id: Chart identifier
            websocket: WebSocket connection
            user_id: User identifier
        """
        if chart_id not in self.websocket_connections:
            self.websocket_connections[chart_id] = []
        
        self.websocket_connections[chart_id].append(websocket)
        
        # Send existing annotations
        chart_annotations = [
            annotation for annotation in self.annotations.values()
            if annotation.chart_id == chart_id and annotation.visible
        ]
        
        await websocket.send_json({
            'type': 'initial_annotations',
            'annotations': [asdict(annotation) for annotation in chart_annotations]
        })
        
        # Notify other users
        await self._notify_user_joined(chart_id, user_id, websocket)
    
    async def disconnect_collaborative_session(self, 
                                             chart_id: str,
                                             websocket: WebSocket,
                                             user_id: str):
        """
        Disconnect user from collaborative session
        
        Args:
            chart_id: Chart identifier
            websocket: WebSocket connection
            user_id: User identifier
        """
        if chart_id in self.websocket_connections:
            try:
                self.websocket_connections[chart_id].remove(websocket)
            except ValueError:
                pass
            
            if not self.websocket_connections[chart_id]:
                del self.websocket_connections[chart_id]
        
        # Notify other users
        await self._notify_user_left(chart_id, user_id)
    
    async def _get_drill_down_data(self, 
                                  drill_path: DrillPath,
                                  chart_config: Dict[str, Any]) -> pd.DataFrame:
        """Get data for drill-down level"""
        # This would integrate with your data source
        # For now, return mock data
        current_level = drill_path.levels[drill_path.current_level]
        
        # Apply filters from drill path
        query_filters = []
        for column, value in drill_path.filters.items():
            query_filters.append(f"{column} = '{value}'")
        
        # Mock implementation
        data = pd.DataFrame({
            current_level: [f"{current_level}_{i}" for i in range(10)],
            'value': np.random.randint(10, 100, 10)
        })
        
        return data
    
    async def _create_drill_down_chart(self, 
                                      data: pd.DataFrame,
                                      drill_path: DrillPath,
                                      chart_config: Dict[str, Any]) -> go.Figure:
        """Create chart for drill-down level"""
        current_level = drill_path.levels[drill_path.current_level]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data[current_level],
            y=data['value'],
            name=current_level.title()
        ))
        
        # Add drill path breadcrumb
        breadcrumb = " > ".join([
            f"{level}: {drill_path.filters.get(level, 'All')}" 
            for level in drill_path.levels[:drill_path.current_level + 1]
        ])
        
        fig.update_layout(
            title=f"Drill Down: {breadcrumb}",
            xaxis_title=current_level.title(),
            yaxis_title="Value"
        )
        
        return fig
    
    async def _apply_filter_to_chart(self, 
                                    chart_id: str,
                                    filter_data: Dict[str, Any],
                                    filter_config: Dict[str, Any]) -> go.Figure:
        """Apply filter to target chart"""
        # This would integrate with your chart data source
        # For now, return a mock filtered chart
        
        fig = go.Figure()
        
        # Mock filtered data
        filtered_values = np.random.randint(1, 50, 10)
        categories = [f"Category {i}" for i in range(10)]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=filtered_values,
            name="Filtered Data"
        ))
        
        fig.update_layout(
            title=f"Filtered by: {filter_data}",
            annotations=[
                dict(
                    text=f"Applied filter: {list(filter_data.keys())[0] if filter_data else 'None'}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,0,0.3)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        return fig
    
    async def _execute_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Execute what-if scenario"""
        if scenario.type == ScenarioType.PARAMETER_CHANGE:
            return await self._execute_parameter_change_scenario(scenario)
        elif scenario.type == ScenarioType.DATA_MODIFICATION:
            return await self._execute_data_modification_scenario(scenario)
        elif scenario.type == ScenarioType.FORECASTING:
            return await self._execute_forecasting_scenario(scenario)
        elif scenario.type == ScenarioType.SENSITIVITY_ANALYSIS:
            return await self._execute_sensitivity_analysis_scenario(scenario)
        else:
            return {'status': 'unsupported_scenario_type'}
    
    async def _execute_parameter_change_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Execute parameter change scenario"""
        # Mock implementation
        baseline_value = 100
        parameter_change = scenario.parameters.get('change_percentage', 0)
        new_value = baseline_value * (1 + parameter_change / 100)
        
        return {
            'scenario_type': 'parameter_change',
            'baseline_value': baseline_value,
            'new_value': new_value,
            'change': new_value - baseline_value,
            'change_percentage': parameter_change
        }
    
    async def _generate_linear_trend_predictions(self, 
                                               data: pd.DataFrame,
                                               periods: int,
                                               confidence_level: float) -> Dict[str, Any]:
        """Generate linear trend predictions"""
        # Simple linear trend implementation
        y = data.iloc[:, 0].values  # Assume first column is the target
        x = np.arange(len(y))
        
        # Fit linear trend
        coeffs = np.polyfit(x, y, 1)
        
        # Generate future predictions
        future_x = np.arange(len(y), len(y) + periods)
        predictions = np.polyval(coeffs, future_x)
        
        # Generate confidence intervals (simplified)
        residuals = y - np.polyval(coeffs, x)
        std_error = np.std(residuals)
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        
        margin_of_error = z_score * std_error
        
        # Create future dates
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'forecast': predictions
        }, index=future_dates)
        
        confidence_df = pd.DataFrame({
            'lower': predictions - margin_of_error,
            'upper': predictions + margin_of_error
        }, index=future_dates)
        
        return {
            'forecast': forecast_df,
            'confidence_intervals': confidence_df,
            'method': 'linear_trend',
            'confidence_level': confidence_level
        }
    
    async def _process_data_for_3d(self, 
                                  data_points: List[Dict[str, Any]],
                                  scene_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process data points for 3D visualization"""
        processed_points = []
        
        for i, point in enumerate(data_points):
            processed_point = {
                'x': point.get('x', 0),
                'y': point.get('y', 0),
                'z': point.get('z', 0),
                'size': point.get('size', 0.5),
                'color': point.get('color', f'hsl({i * 360 / len(data_points)}, 70%, 50%)'),
                'label': point.get('label', f'Point {i}'),
                'data': point
            }
            processed_points.append(processed_point)
        
        return processed_points
    
    async def _cache_drill_path(self, drill_path_id: str, drill_path: DrillPath):
        """Cache drill path in Redis"""
        await self.redis_client.set(
            f"drill_path:{drill_path_id}",
            json.dumps(asdict(drill_path), default=str),
            ex=3600  # 1 hour expiry
        )
    
    async def _cache_annotation(self, annotation_id: str, annotation: Annotation):
        """Cache annotation in Redis"""
        await self.redis_client.set(
            f"annotation:{annotation_id}",
            json.dumps(asdict(annotation), default=str),
            ex=86400  # 24 hour expiry
        )
    
    async def _cache_scenario(self, scenario_id: str, scenario: Scenario):
        """Cache scenario in Redis"""
        await self.redis_client.set(
            f"scenario:{scenario_id}",
            json.dumps(asdict(scenario), default=str),
            ex=86400  # 24 hour expiry
        )
    
    async def _notify_cross_filter_update(self, 
                                         source_chart_id: str,
                                         filter_data: Dict[str, Any],
                                         updated_charts: Dict[str, go.Figure]):
        """Notify clients of cross-filter update"""
        # This would send WebSocket notifications to connected clients
        pass
    
    async def _notify_annotation_added(self, chart_id: str, annotation: Annotation):
        """Notify clients of new annotation"""
        if chart_id in self.websocket_connections:
            message = {
                'type': 'annotation_added',
                'annotation': asdict(annotation)
            }
            
            for websocket in self.websocket_connections[chart_id]:
                try:
                    await websocket.send_json(message)
                except:
                    pass
    
    async def _notify_annotation_updated(self, chart_id: str, annotation: Annotation):
        """Notify clients of annotation update"""
        if chart_id in self.websocket_connections:
            message = {
                'type': 'annotation_updated',
                'annotation': asdict(annotation)
            }
            
            for websocket in self.websocket_connections[chart_id]:
                try:
                    await websocket.send_json(message)
                except:
                    pass
    
    async def _notify_user_joined(self, chart_id: str, user_id: str, websocket: WebSocket):
        """Notify other users that someone joined"""
        if chart_id in self.websocket_connections:
            message = {
                'type': 'user_joined',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            for ws in self.websocket_connections[chart_id]:
                if ws != websocket:  # Don't notify the user who just joined
                    try:
                        await ws.send_json(message)
                    except:
                        pass
    
    async def _notify_user_left(self, chart_id: str, user_id: str):
        """Notify users that someone left"""
        if chart_id in self.websocket_connections:
            message = {
                'type': 'user_left',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            for websocket in self.websocket_connections[chart_id]:
                try:
                    await websocket.send_json(message)
                except:
                    pass
    
    def get_chart_annotations(self, chart_id: str) -> List[Annotation]:
        """Get all annotations for a chart"""
        return [
            annotation for annotation in self.annotations.values()
            if annotation.chart_id == chart_id and annotation.visible
        ]
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID"""
        return self.scenarios.get(scenario_id)
    
    def get_drill_path(self, drill_path_id: str) -> Optional[DrillPath]:
        """Get drill path by ID"""
        return self.drill_paths.get(drill_path_id)