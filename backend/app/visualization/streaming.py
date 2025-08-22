"""
Real-Time Data Streaming Visualization System

Provides real-time data streaming capabilities with WebSocket connections,
Kafka/Pulsar integration, and live dashboard updates.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
import redis
import aiokafka
import aioredis
from collections import deque, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import websockets

from .charts import ChartEngine


class StreamSource(Enum):
    """Types of streaming data sources"""
    KAFKA = "kafka"
    PULSAR = "pulsar"
    REDIS_STREAM = "redis_stream"
    WEBSOCKET = "websocket"
    HTTP_SSE = "http_sse"
    DATABASE_CDC = "database_cdc"
    FILE_TAIL = "file_tail"


class AggregationType(Enum):
    """Types of real-time aggregations"""
    SUM = "sum"
    COUNT = "count"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    LAST = "last"
    FIRST = "first"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    STANDARD_DEVIATION = "std"


class WindowType(Enum):
    """Types of time windows for aggregation"""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    GLOBAL = "global"


@dataclass
class StreamConfig:
    """Configuration for streaming data source"""
    source_type: StreamSource
    connection_params: Dict[str, Any]
    topic_or_key: str
    sampling_rate: Optional[float] = None  # Messages per second
    buffer_size: int = 1000
    timeout_seconds: int = 30


@dataclass
class AggregationConfig:
    """Configuration for real-time aggregation"""
    aggregation_type: AggregationType
    window_type: WindowType
    window_size_seconds: int
    slide_interval_seconds: Optional[int] = None
    group_by_fields: List[str] = None
    filter_expression: Optional[str] = None


@dataclass
class VisualizationConfig:
    """Configuration for real-time visualization"""
    chart_type: str
    update_interval_ms: int = 1000
    max_data_points: int = 100
    auto_scale: bool = True
    color_palette: List[str] = None
    animation_duration: int = 500


@dataclass
class StreamingMetrics:
    """Metrics for streaming data"""
    messages_processed: int = 0
    messages_per_second: float = 0.0
    errors_count: int = 0
    last_message_timestamp: Optional[datetime] = None
    buffer_utilization: float = 0.0
    connection_status: str = "disconnected"


class StreamingVisualizer:
    """
    Real-time data streaming and visualization system
    """
    
    def __init__(self, redis_client: redis.Redis, chart_engine: ChartEngine):
        self.redis_client = redis_client
        self.chart_engine = chart_engine
        
        # Active streams and connections
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        self.data_buffers: Dict[str, deque] = {}
        self.aggregation_windows: Dict[str, Dict[str, Any]] = {}
        
        # Metrics tracking
        self.stream_metrics: Dict[str, StreamingMetrics] = {}
        
        # Background tasks
        self.background_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_stream(self, 
                           stream_id: str,
                           stream_config: StreamConfig,
                           aggregation_config: Optional[AggregationConfig] = None,
                           visualization_config: Optional[VisualizationConfig] = None) -> str:
        """
        Create new streaming data source
        
        Args:
            stream_id: Unique identifier for the stream
            stream_config: Stream configuration
            aggregation_config: Optional aggregation configuration
            visualization_config: Optional visualization configuration
        
        Returns:
            Stream ID
        """
        if stream_id in self.active_streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        # Initialize stream
        self.active_streams[stream_id] = {
            'config': stream_config,
            'aggregation': aggregation_config,
            'visualization': visualization_config,
            'created_at': datetime.utcnow(),
            'status': 'created'
        }
        
        # Initialize data buffer
        self.data_buffers[stream_id] = deque(maxlen=stream_config.buffer_size)
        
        # Initialize metrics
        self.stream_metrics[stream_id] = StreamingMetrics()
        
        # Start background task for data consumption
        task = asyncio.create_task(self._consume_stream_data(stream_id))
        self.background_tasks[stream_id] = task
        
        return stream_id
    
    async def start_stream(self, stream_id: str):
        """
        Start streaming data consumption
        
        Args:
            stream_id: Stream identifier
        """
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        self.active_streams[stream_id]['status'] = 'running'
        self.stream_metrics[stream_id].connection_status = 'connected'
        
        # Start aggregation if configured
        aggregation_config = self.active_streams[stream_id]['aggregation']
        if aggregation_config:
            await self._start_aggregation_window(stream_id, aggregation_config)
    
    async def stop_stream(self, stream_id: str):
        """
        Stop streaming data consumption
        
        Args:
            stream_id: Stream identifier
        """
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['status'] = 'stopped'
            self.stream_metrics[stream_id].connection_status = 'disconnected'
        
        # Cancel background task
        if stream_id in self.background_tasks:
            self.background_tasks[stream_id].cancel()
            del self.background_tasks[stream_id]
    
    async def connect_websocket(self, stream_id: str, websocket: WebSocket):
        """
        Connect WebSocket client for real-time updates
        
        Args:
            stream_id: Stream identifier
            websocket: WebSocket connection
        """
        if stream_id not in self.websocket_connections:
            self.websocket_connections[stream_id] = []
        
        self.websocket_connections[stream_id].append(websocket)
        
        # Send current buffer data
        if stream_id in self.data_buffers:
            buffer_data = list(self.data_buffers[stream_id])
            await websocket.send_json({
                'type': 'initial_data',
                'stream_id': stream_id,
                'data': buffer_data[-50:]  # Send last 50 points
            })
    
    async def disconnect_websocket(self, stream_id: str, websocket: WebSocket):
        """
        Disconnect WebSocket client
        
        Args:
            stream_id: Stream identifier
            websocket: WebSocket connection to remove
        """
        if stream_id in self.websocket_connections:
            try:
                self.websocket_connections[stream_id].remove(websocket)
            except ValueError:
                pass
            
            if not self.websocket_connections[stream_id]:
                del self.websocket_connections[stream_id]
    
    async def get_real_time_chart(self, 
                                 stream_id: str,
                                 chart_type: str = "line",
                                 time_window_minutes: int = 10) -> go.Figure:
        """
        Generate real-time chart from streaming data
        
        Args:
            stream_id: Stream identifier
            chart_type: Type of chart to generate
            time_window_minutes: Time window for data display
        
        Returns:
            Plotly figure with real-time data
        """
        if stream_id not in self.data_buffers:
            raise ValueError(f"Stream {stream_id} not found")
        
        # Get recent data
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=time_window_minutes)
        
        recent_data = []
        for data_point in self.data_buffers[stream_id]:
            if isinstance(data_point, dict) and 'timestamp' in data_point:
                timestamp = datetime.fromisoformat(data_point['timestamp'].replace('Z', '+00:00'))
                if timestamp >= cutoff_time:
                    recent_data.append(data_point)
        
        if not recent_data:
            # Return empty chart
            fig = go.Figure()
            fig.update_layout(title=f"Real-time Data - {stream_id} (No Data)")
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_data)
        
        # Create chart based on type
        if chart_type == "line":
            return self._create_real_time_line_chart(df, stream_id)
        elif chart_type == "scatter":
            return self._create_real_time_scatter_chart(df, stream_id)
        elif chart_type == "bar":
            return self._create_real_time_bar_chart(df, stream_id)
        elif chart_type == "heatmap":
            return self._create_real_time_heatmap(df, stream_id)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    async def get_streaming_metrics(self, stream_id: str) -> StreamingMetrics:
        """
        Get streaming metrics for a stream
        
        Args:
            stream_id: Stream identifier
        
        Returns:
            Current streaming metrics
        """
        if stream_id not in self.stream_metrics:
            raise ValueError(f"Stream {stream_id} not found")
        
        metrics = self.stream_metrics[stream_id]
        
        # Update buffer utilization
        if stream_id in self.data_buffers:
            buffer = self.data_buffers[stream_id]
            metrics.buffer_utilization = len(buffer) / buffer.maxlen
        
        return metrics
    
    async def create_real_time_dashboard(self, 
                                       dashboard_id: str,
                                       stream_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create real-time dashboard with multiple streams
        
        Args:
            dashboard_id: Dashboard identifier
            stream_configs: List of stream configurations
        
        Returns:
            Dashboard configuration
        """
        dashboard_config = {
            'id': dashboard_id,
            'streams': [],
            'layout': {'type': 'grid', 'columns': 2},
            'update_interval_ms': 1000,
            'created_at': datetime.utcnow().isoformat()
        }
        
        for i, config in enumerate(stream_configs):
            stream_id = f"{dashboard_id}_stream_{i}"
            
            # Create stream
            await self.create_stream(
                stream_id=stream_id,
                stream_config=StreamConfig(**config['stream']),
                aggregation_config=AggregationConfig(**config.get('aggregation', {})) if config.get('aggregation') else None,
                visualization_config=VisualizationConfig(**config.get('visualization', {})) if config.get('visualization') else None
            )
            
            dashboard_config['streams'].append({
                'stream_id': stream_id,
                'position': {'x': i % 2, 'y': i // 2, 'width': 1, 'height': 1},
                'chart_type': config.get('chart_type', 'line')
            })
        
        return dashboard_config
    
    async def _consume_stream_data(self, stream_id: str):
        """
        Background task to consume streaming data
        
        Args:
            stream_id: Stream identifier
        """
        stream_config = self.active_streams[stream_id]['config']
        metrics = self.stream_metrics[stream_id]
        
        try:
            if stream_config.source_type == StreamSource.KAFKA:
                await self._consume_kafka_stream(stream_id, stream_config, metrics)
            elif stream_config.source_type == StreamSource.REDIS_STREAM:
                await self._consume_redis_stream(stream_id, stream_config, metrics)
            elif stream_config.source_type == StreamSource.WEBSOCKET:
                await self._consume_websocket_stream(stream_id, stream_config, metrics)
            elif stream_config.source_type == StreamSource.DATABASE_CDC:
                await self._consume_database_cdc(stream_id, stream_config, metrics)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            metrics.errors_count += 1
            metrics.connection_status = "error"
            print(f"Error in stream {stream_id}: {str(e)}")
    
    async def _consume_kafka_stream(self, 
                                   stream_id: str,
                                   config: StreamConfig,
                                   metrics: StreamingMetrics):
        """Consume data from Kafka stream"""
        consumer = aiokafka.AIOKafkaConsumer(
            config.topic_or_key,
            bootstrap_servers=config.connection_params.get('bootstrap_servers', 'localhost:9092'),
            group_id=config.connection_params.get('group_id', f'streaming_viz_{stream_id}'),
            auto_offset_reset='latest'
        )
        
        await consumer.start()
        metrics.connection_status = "connected"
        
        try:
            async for message in consumer:
                if self.active_streams[stream_id]['status'] != 'running':
                    break
                
                # Parse message
                try:
                    data = json.loads(message.value.decode('utf-8'))
                    data['timestamp'] = datetime.utcnow().isoformat()
                    
                    # Add to buffer
                    self.data_buffers[stream_id].append(data)
                    
                    # Update metrics
                    metrics.messages_processed += 1
                    metrics.last_message_timestamp = datetime.utcnow()
                    
                    # Process aggregations
                    await self._process_aggregations(stream_id, data)
                    
                    # Notify WebSocket clients
                    await self._notify_websocket_clients(stream_id, data)
                    
                except json.JSONDecodeError:
                    metrics.errors_count += 1
                
        finally:
            await consumer.stop()
    
    async def _consume_redis_stream(self, 
                                   stream_id: str,
                                   config: StreamConfig,
                                   metrics: StreamingMetrics):
        """Consume data from Redis stream"""
        redis_client = aioredis.from_url(config.connection_params.get('url', 'redis://localhost:6379'))
        
        metrics.connection_status = "connected"
        last_id = '$'  # Start from latest
        
        try:
            while self.active_streams[stream_id]['status'] == 'running':
                try:
                    # Read from stream
                    messages = await redis_client.xread(
                        {config.topic_or_key: last_id},
                        count=10,
                        block=1000
                    )
                    
                    for stream_name, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            # Convert Redis stream format to dict
                            data = {k.decode(): v.decode() for k, v in fields.items()}
                            data['timestamp'] = datetime.utcnow().isoformat()
                            data['message_id'] = message_id.decode()
                            
                            # Add to buffer
                            self.data_buffers[stream_id].append(data)
                            
                            # Update metrics
                            metrics.messages_processed += 1
                            metrics.last_message_timestamp = datetime.utcnow()
                            
                            # Process aggregations
                            await self._process_aggregations(stream_id, data)
                            
                            # Notify WebSocket clients
                            await self._notify_websocket_clients(stream_id, data)
                            
                            last_id = message_id
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    metrics.errors_count += 1
                    await asyncio.sleep(1)
        
        finally:
            await redis_client.close()
    
    async def _consume_websocket_stream(self, 
                                       stream_id: str,
                                       config: StreamConfig,
                                       metrics: StreamingMetrics):
        """Consume data from WebSocket stream"""
        uri = config.connection_params.get('uri')
        
        try:
            async with websockets.connect(uri) as websocket:
                metrics.connection_status = "connected"
                
                async for message in websocket:
                    if self.active_streams[stream_id]['status'] != 'running':
                        break
                    
                    try:
                        data = json.loads(message)
                        data['timestamp'] = datetime.utcnow().isoformat()
                        
                        # Add to buffer
                        self.data_buffers[stream_id].append(data)
                        
                        # Update metrics
                        metrics.messages_processed += 1
                        metrics.last_message_timestamp = datetime.utcnow()
                        
                        # Process aggregations
                        await self._process_aggregations(stream_id, data)
                        
                        # Notify WebSocket clients
                        await self._notify_websocket_clients(stream_id, data)
                        
                    except json.JSONDecodeError:
                        metrics.errors_count += 1
        
        except Exception as e:
            metrics.connection_status = "error"
            metrics.errors_count += 1
    
    async def _consume_database_cdc(self, 
                                   stream_id: str,
                                   config: StreamConfig,
                                   metrics: StreamingMetrics):
        """Consume database change data capture events"""
        # This would integrate with database CDC systems like Debezium
        # Implementation depends on specific CDC system
        pass
    
    async def _process_aggregations(self, stream_id: str, data: Dict[str, Any]):
        """Process real-time aggregations for streaming data"""
        aggregation_config = self.active_streams[stream_id]['aggregation']
        if not aggregation_config:
            return
        
        # Initialize aggregation window if needed
        if stream_id not in self.aggregation_windows:
            self.aggregation_windows[stream_id] = {
                'data': deque(),
                'last_slide': datetime.utcnow()
            }
        
        window = self.aggregation_windows[stream_id]
        current_time = datetime.utcnow()
        
        # Add data to window
        window['data'].append({
            'timestamp': current_time,
            'data': data
        })
        
        # Remove old data based on window type
        if aggregation_config.window_type == WindowType.TUMBLING:
            await self._process_tumbling_window(stream_id, aggregation_config, window, current_time)
        elif aggregation_config.window_type == WindowType.SLIDING:
            await self._process_sliding_window(stream_id, aggregation_config, window, current_time)
    
    async def _process_tumbling_window(self, 
                                      stream_id: str,
                                      config: AggregationConfig,
                                      window: Dict[str, Any],
                                      current_time: datetime):
        """Process tumbling window aggregation"""
        window_start = window['last_slide']
        window_end = window_start + timedelta(seconds=config.window_size_seconds)
        
        if current_time >= window_end:
            # Calculate aggregation for completed window
            window_data = [item['data'] for item in window['data'] 
                          if window_start <= item['timestamp'] < window_end]
            
            if window_data:
                aggregated = await self._calculate_aggregation(window_data, config)
                
                # Store aggregation result
                aggregation_result = {
                    'timestamp': window_end.isoformat(),
                    'window_start': window_start.isoformat(),
                    'window_end': window_end.isoformat(),
                    'aggregation': aggregated,
                    'count': len(window_data)
                }
                
                # Notify WebSocket clients of aggregation
                await self._notify_websocket_clients(stream_id, aggregation_result, message_type='aggregation')
            
            # Update window
            window['last_slide'] = window_end
            window['data'] = deque([item for item in window['data'] if item['timestamp'] >= window_end])
    
    async def _process_sliding_window(self, 
                                     stream_id: str,
                                     config: AggregationConfig,
                                     window: Dict[str, Any],
                                     current_time: datetime):
        """Process sliding window aggregation"""
        slide_interval = config.slide_interval_seconds or config.window_size_seconds
        
        if (current_time - window['last_slide']).total_seconds() >= slide_interval:
            # Calculate aggregation for current window
            window_start = current_time - timedelta(seconds=config.window_size_seconds)
            window_data = [item['data'] for item in window['data'] 
                          if item['timestamp'] >= window_start]
            
            if window_data:
                aggregated = await self._calculate_aggregation(window_data, config)
                
                # Store aggregation result
                aggregation_result = {
                    'timestamp': current_time.isoformat(),
                    'window_start': window_start.isoformat(),
                    'window_end': current_time.isoformat(),
                    'aggregation': aggregated,
                    'count': len(window_data)
                }
                
                # Notify WebSocket clients of aggregation
                await self._notify_websocket_clients(stream_id, aggregation_result, message_type='aggregation')
            
            # Update last slide time
            window['last_slide'] = current_time
            
            # Clean old data
            cutoff_time = current_time - timedelta(seconds=config.window_size_seconds * 2)
            window['data'] = deque([item for item in window['data'] if item['timestamp'] >= cutoff_time])
    
    async def _calculate_aggregation(self, 
                                   data: List[Dict[str, Any]],
                                   config: AggregationConfig) -> Dict[str, Any]:
        """Calculate aggregation based on configuration"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        # Group by fields if specified
        if config.group_by_fields:
            grouped = df.groupby(config.group_by_fields)
            result = {}
            
            for group_name, group_data in grouped:
                group_key = group_name if isinstance(group_name, str) else '_'.join(map(str, group_name))
                result[group_key] = self._apply_aggregation_function(group_data, config.aggregation_type)
            
            return result
        else:
            return self._apply_aggregation_function(df, config.aggregation_type)
    
    def _apply_aggregation_function(self, 
                                   df: pd.DataFrame,
                                   aggregation_type: AggregationType) -> Dict[str, Any]:
        """Apply aggregation function to DataFrame"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if aggregation_type == AggregationType.SUM:
            return df[numeric_columns].sum().to_dict()
        elif aggregation_type == AggregationType.COUNT:
            return {'count': len(df)}
        elif aggregation_type == AggregationType.AVERAGE:
            return df[numeric_columns].mean().to_dict()
        elif aggregation_type == AggregationType.MIN:
            return df[numeric_columns].min().to_dict()
        elif aggregation_type == AggregationType.MAX:
            return df[numeric_columns].max().to_dict()
        elif aggregation_type == AggregationType.MEDIAN:
            return df[numeric_columns].median().to_dict()
        elif aggregation_type == AggregationType.STANDARD_DEVIATION:
            return df[numeric_columns].std().to_dict()
        else:
            return {}
    
    async def _notify_websocket_clients(self, 
                                       stream_id: str,
                                       data: Dict[str, Any],
                                       message_type: str = 'data'):
        """Notify WebSocket clients of new data"""
        if stream_id not in self.websocket_connections:
            return
        
        message = {
            'type': message_type,
            'stream_id': stream_id,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections[stream_id]:
            try:
                await websocket.send_json(message)
            except:
                disconnected_clients.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected_clients:
            await self.disconnect_websocket(stream_id, websocket)
    
    def _create_real_time_line_chart(self, df: pd.DataFrame, stream_id: str) -> go.Figure:
        """Create real-time line chart"""
        fig = go.Figure()
        
        # Assume timestamp column exists
        if 'timestamp' in df.columns:
            x_values = pd.to_datetime(df['timestamp'])
            
            # Plot numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=f'Real-time Data - {stream_id}',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def _create_real_time_scatter_chart(self, df: pd.DataFrame, stream_id: str) -> go.Figure:
        """Create real-time scatter chart"""
        fig = go.Figure()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            fig.add_trace(go.Scatter(
                x=df[numeric_columns[0]],
                y=df[numeric_columns[1]],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df.index,
                    colorscale='viridis',
                    showscale=True
                ),
                name='Data Points'
            ))
        
        fig.update_layout(
            title=f'Real-time Scatter - {stream_id}',
            xaxis_title=numeric_columns[0] if len(numeric_columns) > 0 else 'X',
            yaxis_title=numeric_columns[1] if len(numeric_columns) > 1 else 'Y'
        )
        
        return fig
    
    def _create_real_time_bar_chart(self, df: pd.DataFrame, stream_id: str) -> go.Figure:
        """Create real-time bar chart"""
        fig = go.Figure()
        
        # Aggregate recent data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0 and len(numeric_columns) > 0:
            grouped = df.groupby(categorical_columns[0])[numeric_columns[0]].sum()
            
            fig.add_trace(go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=numeric_columns[0]
            ))
        
        fig.update_layout(
            title=f'Real-time Bar Chart - {stream_id}',
            xaxis_title='Category',
            yaxis_title='Value'
        )
        
        return fig
    
    def _create_real_time_heatmap(self, df: pd.DataFrame, stream_id: str) -> go.Figure:
        """Create real-time heatmap"""
        fig = go.Figure()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            # Create correlation heatmap
            correlation_matrix = df[numeric_columns].corr()
            
            fig.add_trace(go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='viridis'
            ))
        
        fig.update_layout(
            title=f'Real-time Heatmap - {stream_id}'
        )
        
        return fig
    
    async def _start_aggregation_window(self, stream_id: str, config: AggregationConfig):
        """Initialize aggregation window for stream"""
        self.aggregation_windows[stream_id] = {
            'data': deque(),
            'last_slide': datetime.utcnow(),
            'config': config
        }
    
    def get_active_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active streams"""
        return {
            stream_id: {
                'config': asdict(stream_info['config']),
                'status': stream_info['status'],
                'created_at': stream_info['created_at'].isoformat(),
                'metrics': asdict(self.stream_metrics.get(stream_id, StreamingMetrics()))
            }
            for stream_id, stream_info in self.active_streams.items()
        }