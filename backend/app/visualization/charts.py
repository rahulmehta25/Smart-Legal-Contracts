"""
Advanced Charting Engine

Provides comprehensive visualization capabilities using D3.js, Plotly, and Apache Superset integration.
Supports interactive, 3D, and real-time visualizations for legal contract analysis.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import networkx as nx
from wordcloud import WordCloud
import squarify
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


@dataclass
class ChartConfig:
    """Configuration for chart generation"""
    chart_type: str
    title: str
    width: int = 800
    height: int = 600
    theme: str = 'plotly_white'
    interactive: bool = True
    export_format: str = 'html'
    color_palette: List[str] = None


class ChartEngine:
    """
    Advanced charting engine with support for complex visualizations
    """
    
    def __init__(self):
        self.default_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def create_3d_clause_relationship_graph(self, 
                                          clause_data: Dict[str, Any],
                                          relationships: List[Dict]) -> go.Figure:
        """
        Create 3D graph showing relationships between contract clauses
        
        Args:
            clause_data: Dictionary containing clause information
            relationships: List of relationships between clauses
        
        Returns:
            Plotly 3D scatter plot figure
        """
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for clause_id, clause_info in clause_data.items():
            G.add_node(clause_id, **clause_info)
        
        # Add edges
        for rel in relationships:
            G.add_edge(rel['source'], rel['target'], weight=rel.get('strength', 1))
        
        # Calculate 3D layout using spring layout
        pos = nx.spring_layout(G, dim=3)
        
        # Extract coordinates
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        
        # Create edges
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.5)', width=2),
            hoverinfo='none',
            name='Relationships'
        ))
        
        # Add nodes
        node_colors = [clause_data[node].get('risk_level', 0.5) for node in G.nodes()]
        node_text = [f"{node}: {clause_data[node].get('type', 'Unknown')}" for node in G.nodes()]
        
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title="Risk Level"),
                line=dict(width=2)
            ),
            text=list(G.nodes()),
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            name='Clauses'
        ))
        
        fig.update_layout(
            title='3D Clause Relationship Network',
            scene=dict(
                xaxis_title='X Dimension',
                yaxis_title='Y Dimension',
                zaxis_title='Z Dimension',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            showlegend=True,
            width=1000,
            height=800
        )
        
        return fig
    
    def create_geographical_risk_heatmap(self, 
                                       risk_data: List[Dict],
                                       locations: Dict[str, Tuple[float, float]]) -> go.Figure:
        """
        Create geographical heatmap showing legal risks by jurisdiction
        
        Args:
            risk_data: List of risk assessments with location data
            locations: Dictionary mapping location names to coordinates
        
        Returns:
            Plotly map figure with risk heatmap
        """
        # Prepare data for mapping
        lats, lons, risks, hover_texts = [], [], [], []
        
        for risk_item in risk_data:
            location = risk_item.get('jurisdiction')
            if location in locations:
                lat, lon = locations[location]
                lats.append(lat)
                lons.append(lon)
                risks.append(risk_item.get('risk_score', 0))
                hover_texts.append(
                    f"Location: {location}<br>"
                    f"Risk Score: {risk_item.get('risk_score', 0):.2f}<br>"
                    f"Cases: {risk_item.get('case_count', 0)}<br>"
                    f"Primary Risk: {risk_item.get('primary_risk_factor', 'Unknown')}"
                )
        
        # Create heatmap
        fig = go.Figure(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=[max(10, r * 50) for r in risks],
                color=risks,
                colorscale='RdYlBu_r',
                colorbar=dict(title="Risk Level"),
                sizemode='diameter',
                opacity=0.7
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Risk Locations'
        ))
        
        fig.update_layout(
            title='Geographical Legal Risk Assessment',
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=39.8283, lon=-98.5795),  # US center
                zoom=3
            ),
            width=1200,
            height=800
        )
        
        return fig
    
    def create_time_series_analysis(self, 
                                  time_data: pd.DataFrame,
                                  metrics: List[str]) -> go.Figure:
        """
        Create comprehensive time series analysis with multiple metrics
        
        Args:
            time_data: DataFrame with time-indexed data
            metrics: List of metric columns to plot
        
        Returns:
            Plotly subplot figure with time series
        """
        # Create subplots
        rows = len(metrics)
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=metrics,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        colors = self.default_palette[:len(metrics)]
        
        for i, metric in enumerate(metrics, 1):
            if metric in time_data.columns:
                # Add main series
                fig.add_trace(
                    go.Scatter(
                        x=time_data.index,
                        y=time_data[metric],
                        mode='lines',
                        name=metric,
                        line=dict(color=colors[i-1], width=2),
                        hovertemplate=f"{metric}: %{{y:.2f}}<extra></extra>"
                    ),
                    row=i, col=1
                )
                
                # Add trend line
                trend = self._calculate_trend(time_data[metric])
                fig.add_trace(
                    go.Scatter(
                        x=time_data.index,
                        y=trend,
                        mode='lines',
                        name=f'{metric} Trend',
                        line=dict(color=colors[i-1], width=1, dash='dash'),
                        opacity=0.6
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title='Time Series Analysis Dashboard',
            height=200 * rows + 100,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_sankey_clause_flow(self, flow_data: List[Dict]) -> go.Figure:
        """
        Create Sankey diagram showing clause flow through contract lifecycle
        
        Args:
            flow_data: List of flow relationships
        
        Returns:
            Plotly Sankey diagram
        """
        # Extract unique nodes
        nodes = set()
        for flow in flow_data:
            nodes.add(flow['source'])
            nodes.add(flow['target'])
        
        node_list = list(nodes)
        node_dict = {node: i for i, node in enumerate(node_list)}
        
        # Prepare Sankey data
        source_indices = [node_dict[flow['source']] for flow in flow_data]
        target_indices = [node_dict[flow['target']] for flow in flow_data]
        values = [flow.get('value', 1) for flow in flow_data]
        
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_list,
                color=self.default_palette[:len(node_list)]
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=['rgba(31,119,180,0.4)'] * len(values)
            )
        ))
        
        fig.update_layout(
            title="Contract Clause Flow Analysis",
            font_size=12,
            width=1200,
            height=600
        )
        
        return fig
    
    def create_network_entity_graph(self, 
                                   entities: Dict[str, Any],
                                   relationships: List[Dict]) -> go.Figure:
        """
        Create network graph showing entity relationships
        
        Args:
            entities: Dictionary of entities with properties
            relationships: List of relationships between entities
        
        Returns:
            Plotly network graph
        """
        G = nx.Graph()
        
        # Add nodes
        for entity_id, entity_data in entities.items():
            G.add_node(entity_id, **entity_data)
        
        # Add edges
        for rel in relationships:
            G.add_edge(rel['source'], rel['target'], 
                      weight=rel.get('strength', 1),
                      type=rel.get('type', 'unknown'))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create traces for edges
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge[2].get('weight', 1), color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create trace for nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=20,
                color=[entities[node].get('category_color', '#1f77b4') for node in G.nodes()],
                line=dict(width=2)
            ),
            text=list(G.nodes()),
            textposition="middle center"
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Entity Relationship Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Entity relationships in contract network",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="#888", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_word_cloud_visualization(self, text_data: List[str]) -> Dict[str, Any]:
        """
        Create word cloud for key terms analysis
        
        Args:
            text_data: List of text strings to analyze
        
        Returns:
            Dictionary containing word cloud data and metadata
        """
        # Combine all text
        combined_text = ' '.join(text_data)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5
        ).generate(combined_text)
        
        # Extract word frequencies
        frequencies = wordcloud.words_
        
        return {
            'wordcloud_image': wordcloud.to_image(),
            'word_frequencies': frequencies,
            'top_words': sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:20],
            'total_words': len(frequencies),
            'text_length': len(combined_text)
        }
    
    def create_treemap_hierarchical(self, hierarchical_data: Dict[str, Any]) -> go.Figure:
        """
        Create treemap for hierarchical data visualization
        
        Args:
            hierarchical_data: Nested dictionary representing hierarchy
        
        Returns:
            Plotly treemap figure
        """
        # Flatten hierarchical data
        labels, parents, values, colors = [], [], [], []
        
        def flatten_hierarchy(data, parent='', level=0):
            for key, value in data.items():
                labels.append(key)
                parents.append(parent)
                
                if isinstance(value, dict):
                    if 'value' in value:
                        values.append(value['value'])
                        colors.append(value.get('color', level))
                    else:
                        values.append(0)
                        colors.append(level)
                    
                    # Recursively process children
                    if 'children' in value:
                        flatten_hierarchy(value['children'], key, level + 1)
                else:
                    values.append(value if isinstance(value, (int, float)) else 1)
                    colors.append(level)
        
        flatten_hierarchy(hierarchical_data)
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percent: %{percentParent}<extra></extra>',
            maxdepth=4
        ))
        
        fig.update_layout(
            title="Hierarchical Data Analysis",
            width=1200,
            height=800
        )
        
        return fig
    
    def create_predictive_visualization(self, 
                                      historical_data: pd.DataFrame,
                                      predictions: pd.DataFrame,
                                      confidence_intervals: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create predictive visualization with confidence intervals
        
        Args:
            historical_data: Historical time series data
            predictions: Future predictions
            confidence_intervals: Optional confidence interval data
        
        Returns:
            Plotly figure with predictions
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.iloc[:, 0],
            mode='lines',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions.iloc[:, 0],
            mode='lines',
            name='Predictions',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            fig.add_trace(go.Scatter(
                x=predictions.index,
                y=confidence_intervals.iloc[:, 1],  # Upper bound
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions.index,
                y=confidence_intervals.iloc[:, 0],  # Lower bound
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,127,14,0.2)',
                fill='tonexty',
                name='Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title='Predictive Analysis with Confidence Intervals',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        return fig
    
    def _calculate_trend(self, series: pd.Series) -> np.ndarray:
        """Calculate trend line for time series"""
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.fillna(series.mean()), 1)
        return np.polyval(coeffs, x)
    
    def export_chart(self, fig: go.Figure, format: str = 'html', filename: str = None) -> str:
        """
        Export chart in specified format
        
        Args:
            fig: Plotly figure to export
            format: Export format ('html', 'png', 'pdf', 'svg')
            filename: Optional filename
        
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chart_{timestamp}"
        
        filepath = f"/tmp/{filename}.{format}"
        
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath, format='png', width=1200, height=800)
        elif format == 'pdf':
            fig.write_image(filepath, format='pdf', width=1200, height=800)
        elif format == 'svg':
            fig.write_image(filepath, format='svg', width=1200, height=800)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return filepath
    
    def create_ar_vr_data_scene(self, data_points: List[Dict]) -> Dict[str, Any]:
        """
        Prepare data for AR/VR visualization using A-Frame format
        
        Args:
            data_points: List of 3D data points with properties
        
        Returns:
            Dictionary containing A-Frame scene configuration
        """
        scene_config = {
            "a-scene": {
                "embedded": True,
                "vr-mode-ui": {"enabled": True},
                "entities": []
            }
        }
        
        # Add data points as 3D entities
        for i, point in enumerate(data_points):
            entity = {
                "a-sphere": {
                    "position": f"{point.get('x', 0)} {point.get('y', 0)} {point.get('z', 0)}",
                    "radius": point.get('size', 0.5),
                    "color": point.get('color', '#1f77b4'),
                    "animation": {
                        "property": "rotation",
                        "to": "0 360 0",
                        "loop": True,
                        "dur": 10000
                    },
                    "text": {
                        "value": point.get('label', f'Point {i}'),
                        "position": "0 1 0",
                        "align": "center"
                    }
                }
            }
            scene_config["a-scene"]["entities"].append(entity)
        
        # Add lighting and camera
        scene_config["a-scene"]["entities"].extend([
            {"a-light": {"type": "ambient", "color": "#404040"}},
            {"a-light": {"type": "point", "position": "0 10 0", "color": "#ffffff"}},
            {"a-camera": {
                "position": "0 2 5",
                "look-controls": True,
                "wasd-controls": True
            }}
        ])
        
        return scene_config