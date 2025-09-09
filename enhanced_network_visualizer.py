#!/usr/bin/env python3
"""
Enhanced Network Visualization System
Advanced visualization capabilities for cryptocurrency transaction analysis

This module provides:
- Interactive 3D network visualization
- Advanced filtering and clustering
- Real-time risk monitoring
- Export capabilities for law enforcement
- Multi-blockchain support
- Temporal analysis visualization
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import json
import logging
from dataclasses import dataclass
from enum import Enum
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VisualizationMode(Enum):
    """Different visualization modes"""
    NETWORK_2D = "network_2d"
    NETWORK_3D = "network_3d"
    TEMPORAL = "temporal"
    RISK_HEATMAP = "risk_heatmap"
    COMMUNITY_ANALYSIS = "community_analysis"
    TRANSACTION_FLOW = "transaction_flow"

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    mode: VisualizationMode
    show_risk_colors: bool = True
    show_communities: bool = True
    show_temporal_animation: bool = False
    risk_threshold: float = 0.5
    max_nodes: int = 1000
    max_edges: int = 5000
    animation_speed: float = 1.0
    color_scheme: str = "viridis"

class EnhancedNetworkVisualizer:
    """
    Advanced network visualization system for cryptocurrency analysis
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """Initialize the enhanced visualizer"""
        self.config = config or VisualizationConfig(VisualizationMode.NETWORK_2D)
        self.color_palettes = {
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma,
            'risk': ['#00ff00', '#ffff00', '#ff8000', '#ff0000', '#800000']
        }
        
        logger.info("Enhanced Network Visualizer initialized")

    def create_interactive_3d_network(self, graph: nx.DiGraph, risk_profiles: Dict[str, Dict], 
                                   communities: Dict[str, int] = None) -> go.Figure:
        """
        Create interactive 3D network visualization with advanced features
        """
        logger.info("Creating interactive 3D network visualization")
        
        # Prepare node data
        node_data = self._prepare_node_data(graph, risk_profiles, communities)
        edge_data = self._prepare_edge_data(graph)
        
        # Create 3D scatter plot for nodes
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=node_data['x'],
            y=node_data['y'],
            z=node_data['z'],
            mode='markers',
            marker=dict(
                size=node_data['size'],
                color=node_data['color'],
                colorscale=self.color_palettes[self.config.color_scheme],
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            text=node_data['text'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Risk Level: %{customdata[0]}<br>' +
                         'Risk Score: %{customdata[1]:.3f}<br>' +
                         'Transactions: %{customdata[2]}<br>' +
                         'Volume: %{customdata[3]:.2f} BTC<br>' +
                         '<extra></extra>',
            customdata=node_data['customdata'],
            name='Addresses'
        ))
        
        # Add edges
        if edge_data['x']:
            fig.add_trace(go.Scatter3d(
                x=edge_data['x'],
                y=edge_data['y'],
                z=edge_data['z'],
                mode='lines',
                line=dict(color='rgba(100,100,100,0.3)', width=1),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='3D Cryptocurrency Transaction Network',
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Risk Level',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        logger.info("3D network visualization created")
        return fig

    def create_temporal_analysis(self, transactions: List[Dict], risk_profiles: Dict[str, Dict]) -> go.Figure:
        """
        Create temporal analysis visualization showing transaction patterns over time
        """
        logger.info("Creating temporal analysis visualization")
        
        # Convert transactions to DataFrame
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Transaction Volume Over Time', 'Risk Score Distribution',
                          'Transaction Count by Hour', 'Risk Level Timeline',
                          'Address Activity Heatmap', 'Suspicious Pattern Detection'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Transaction volume over time
        daily_volume = df.groupby(df['timestamp'].dt.date)['value'].sum()
        fig.add_trace(
            go.Scatter(x=daily_volume.index, y=daily_volume.values, 
                      mode='lines+markers', name='Daily Volume'),
            row=1, col=1
        )
        
        # Risk score distribution
        risk_scores = [profile['risk_score'] for profile in risk_profiles.values()]
        fig.add_trace(
            go.Histogram(x=risk_scores, nbinsx=20, name='Risk Score Distribution'),
            row=1, col=2
        )
        
        # Transaction count by hour
        hourly_count = df.groupby(df['timestamp'].dt.hour).size()
        fig.add_trace(
            go.Bar(x=hourly_count.index, y=hourly_count.values, name='Hourly Transactions'),
            row=2, col=1
        )
        
        # Risk level timeline
        risk_timeline = self._create_risk_timeline(df, risk_profiles)
        fig.add_trace(
            go.Scatter(x=risk_timeline['timestamp'], y=risk_timeline['risk_level'],
                      mode='markers', name='Risk Timeline'),
            row=2, col=2
        )
        
        # Address activity heatmap
        activity_matrix = self._create_activity_heatmap(df)
        fig.add_trace(
            go.Heatmap(z=activity_matrix.values, 
                      x=activity_matrix.columns,
                      y=activity_matrix.index,
                      colorscale='Viridis'),
            row=3, col=1
        )
        
        # Suspicious pattern detection
        pattern_timeline = self._create_pattern_timeline(df)
        fig.add_trace(
            go.Scatter(x=pattern_timeline['timestamp'], y=pattern_timeline['pattern_count'],
                      mode='lines+markers', name='Suspicious Patterns'),
            row=3, col=2
        )
        
        fig.update_layout(
            title='Temporal Analysis of Cryptocurrency Transactions',
            height=1200,
            showlegend=True
        )
        
        logger.info("Temporal analysis visualization created")
        return fig

    def create_risk_heatmap(self, risk_profiles: Dict[str, Dict], 
                          interaction_matrix: np.ndarray = None) -> go.Figure:
        """
        Create risk heatmap showing risk interactions between addresses
        """
        logger.info("Creating risk heatmap visualization")
        
        addresses = list(risk_profiles.keys())
        n_addresses = len(addresses)
        
        # Create risk matrix
        risk_matrix = np.zeros((n_addresses, n_addresses))
        
        for i, addr1 in enumerate(addresses):
            for j, addr2 in enumerate(addresses):
                if i == j:
                    # Diagonal: individual risk score
                    risk_matrix[i][j] = risk_profiles[addr1]['risk_score']
                else:
                    # Off-diagonal: interaction risk
                    if interaction_matrix is not None:
                        risk_matrix[i][j] = interaction_matrix[i][j]
                    else:
                        # Simple interaction risk calculation
                        risk_matrix[i][j] = (risk_profiles[addr1]['risk_score'] + 
                                           risk_profiles[addr2]['risk_score']) / 2
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=addresses,
            y=addresses,
            colorscale='Reds',
            hoverongaps=False,
            hovertemplate='Address 1: %{y}<br>Address 2: %{x}<br>Risk Score: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk Interaction Heatmap',
            xaxis_title='Addresses',
            yaxis_title='Addresses',
            width=1000,
            height=800
        )
        
        logger.info("Risk heatmap visualization created")
        return fig

    def create_community_analysis(self, graph: nx.DiGraph, communities: Dict[str, int], 
                                risk_profiles: Dict[str, Dict]) -> go.Figure:
        """
        Create community analysis visualization
        """
        logger.info("Creating community analysis visualization")
        
        # Prepare community data
        community_data = self._prepare_community_data(graph, communities, risk_profiles)
        
        # Create subplots for community analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Community Size Distribution', 'Community Risk Levels',
                          'Inter-Community Connections', 'Community Activity Timeline'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Community size distribution
        community_sizes = community_data['sizes']
        fig.add_trace(
            go.Bar(x=list(community_sizes.keys()), y=list(community_sizes.values()),
                  name='Community Sizes'),
            row=1, col=1
        )
        
        # Community risk levels
        community_risks = community_data['risks']
        fig.add_trace(
            go.Bar(x=list(community_risks.keys()), y=list(community_risks.values()),
                  name='Community Risk Levels'),
            row=1, col=2
        )
        
        # Inter-community connections
        inter_connections = community_data['inter_connections']
        fig.add_trace(
            go.Scatter(x=inter_connections['x'], y=inter_connections['y'],
                      mode='markers', name='Inter-Community Connections'),
            row=2, col=1
        )
        
        # Community activity timeline
        activity_timeline = community_data['activity_timeline']
        fig.add_trace(
            go.Scatter(x=activity_timeline['timestamp'], y=activity_timeline['activity'],
                      mode='lines+markers', name='Community Activity'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Community Analysis',
            height=800,
            showlegend=True
        )
        
        logger.info("Community analysis visualization created")
        return fig

    def create_transaction_flow_diagram(self, transactions: List[Dict], 
                                      risk_profiles: Dict[str, Dict]) -> go.Figure:
        """
        Create transaction flow diagram showing money movement patterns
        """
        logger.info("Creating transaction flow diagram")
        
        # Create Sankey diagram for transaction flows
        source_indices = []
        target_indices = []
        values = []
        labels = []
        colors = []
        
        # Prepare data for Sankey diagram
        address_to_index = {}
        index = 0
        
        for tx in transactions:
            from_addr = tx.get('from_address', '')
            to_addr = tx.get('to_address', '')
            value = tx.get('value', 0)
            
            if from_addr not in address_to_index:
                address_to_index[from_addr] = index
                labels.append(from_addr[:10] + '...')
                # Color based on risk level
                risk_level = risk_profiles.get(from_addr, {}).get('risk_level', 'CLEAN')
                colors.append(self._get_risk_color(risk_level))
                index += 1
            
            if to_addr not in address_to_index:
                address_to_index[to_addr] = index
                labels.append(to_addr[:10] + '...')
                risk_level = risk_profiles.get(to_addr, {}).get('risk_level', 'CLEAN')
                colors.append(self._get_risk_color(risk_level))
                index += 1
            
            source_indices.append(address_to_index[from_addr])
            target_indices.append(address_to_index[to_addr])
            values.append(value)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=colors
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color='rgba(100,100,100,0.3)'
            )
        )])
        
        fig.update_layout(
            title="Transaction Flow Diagram",
            font_size=10,
            width=1200,
            height=600
        )
        
        logger.info("Transaction flow diagram created")
        return fig

    def create_law_enforcement_dashboard(self, analysis_results: Dict) -> dash.Dash:
        """
        Create comprehensive law enforcement dashboard
        """
        logger.info("Creating law enforcement dashboard")
        
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Dashboard layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚨 ChainBreak Law Enforcement Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Summary cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis_results['summary']['critical_risk_addresses']}", 
                                   className="text-danger"),
                            html.P("Critical Risk Addresses", className="card-text")
                        ])
                    ], color="danger", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis_results['summary']['high_risk_addresses']}", 
                                   className="text-warning"),
                            html.P("High Risk Addresses", className="card-text")
                        ])
                    ], color="warning", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis_results['summary']['total_patterns_detected']}", 
                                   className="text-info"),
                            html.P("Suspicious Patterns", className="card-text")
                        ])
                    ], color="info", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis_results['summary']['total_addresses_analyzed']}", 
                                   className="text-primary"),
                            html.P("Total Addresses", className="card-text")
                        ])
                    ], color="primary", outline=True)
                ], width=3)
            ], className="mb-4"),
            
            # Visualization tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id='network-graph')
                        ], label="Network Analysis", tab_id="network"),
                        dbc.Tab([
                            dcc.Graph(id='temporal-analysis')
                        ], label="Temporal Analysis", tab_id="temporal"),
                        dbc.Tab([
                            dcc.Graph(id='risk-heatmap')
                        ], label="Risk Heatmap", tab_id="risk"),
                        dbc.Tab([
                            dcc.Graph(id='community-analysis')
                        ], label="Community Analysis", tab_id="community"),
                        dbc.Tab([
                            dcc.Graph(id='transaction-flow')
                        ], label="Transaction Flow", tab_id="flow")
                    ], id="tabs", active_tab="network")
                ], width=12)
            ]),
            
            # Alerts and recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🚨 Critical Alerts"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li(alert) for alert in analysis_results['recommendations']
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Top Risk Addresses"),
                        dbc.CardBody([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Address"),
                                        html.Th("Risk Level"),
                                        html.Th("Score")
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(addr['address'][:20] + '...'),
                                        html.Td(addr['risk_level']),
                                        html.Td(f"{addr['risk_score']:.3f}")
                                    ]) for addr in analysis_results['top_risk_addresses'][:5]
                                ])
                            ], className="table table-sm")
                        ])
                    ])
                ], width=6)
            ], className="mt-4")
        ], fluid=True)
        
        # Callbacks for interactive features
        @app.callback(
            Output('network-graph', 'figure'),
            Input('tabs', 'active_tab')
        )
        def update_network_graph(active_tab):
            if active_tab == 'network':
                # Create network visualization
                return self.create_interactive_3d_network(
                    analysis_results['graph'], 
                    analysis_results['risk_profiles']
                )
            return {}
        
        logger.info("Law enforcement dashboard created")
        return app

    def _prepare_node_data(self, graph: nx.DiGraph, risk_profiles: Dict[str, Dict], 
                          communities: Dict[str, int] = None) -> Dict:
        """Prepare node data for visualization"""
        nodes = list(graph.nodes())
        
        # Calculate 3D positions using spring layout
        pos_2d = nx.spring_layout(graph, dim=2)
        pos_3d = {}
        
        for node in nodes:
            x, y = pos_2d[node]
            z = risk_profiles.get(node, {}).get('risk_score', 0.0)
            pos_3d[node] = (x, y, z)
        
        node_data = {
            'x': [pos_3d[node][0] for node in nodes],
            'y': [pos_3d[node][1] for node in nodes],
            'z': [pos_3d[node][2] for node in nodes],
            'size': [self._calculate_node_size(node, risk_profiles) for node in nodes],
            'color': [risk_profiles.get(node, {}).get('risk_score', 0.0) for node in nodes],
            'text': [f"{node[:10]}..." for node in nodes],
            'customdata': [[
                risk_profiles.get(node, {}).get('risk_level', 'CLEAN'),
                risk_profiles.get(node, {}).get('risk_score', 0.0),
                risk_profiles.get(node, {}).get('transaction_count', 0),
                risk_profiles.get(node, {}).get('transaction_volume', 0.0)
            ] for node in nodes]
        }
        
        return node_data

    def _prepare_edge_data(self, graph: nx.DiGraph) -> Dict:
        """Prepare edge data for visualization"""
        edges = list(graph.edges())
        
        if not edges:
            return {'x': [], 'y': [], 'z': []}
        
        # Get node positions (simplified)
        pos = nx.spring_layout(graph, dim=3)
        
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in edges:
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        return {'x': edge_x, 'y': edge_y, 'z': edge_z}

    def _calculate_node_size(self, node: str, risk_profiles: Dict[str, Dict]) -> float:
        """Calculate node size based on risk and activity"""
        profile = risk_profiles.get(node, {})
        base_size = 10
        risk_multiplier = 1 + profile.get('risk_score', 0.0) * 2
        volume_multiplier = 1 + min(profile.get('transaction_volume', 0.0) / 100000000, 2.0)
        
        return base_size * risk_multiplier * volume_multiplier

    def _get_risk_color(self, risk_level: str) -> str:
        """Get color based on risk level"""
        color_map = {
            'CRITICAL': '#ff0000',
            'HIGH': '#ff8000',
            'MEDIUM': '#ffff00',
            'LOW': '#80ff00',
            'CLEAN': '#00ff00'
        }
        return color_map.get(risk_level, '#808080')

    def _create_risk_timeline(self, df: pd.DataFrame, risk_profiles: Dict[str, Dict]) -> Dict:
        """Create risk timeline data"""
        # Simplified implementation
        return {
            'timestamp': df['timestamp'].tolist(),
            'risk_level': [risk_profiles.get(addr, {}).get('risk_score', 0.0) 
                          for addr in df['from_address'].tolist()]
        }

    def _create_activity_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create activity heatmap data"""
        # Group by hour and address
        df['hour'] = df['timestamp'].dt.hour
        activity_matrix = df.groupby(['hour', 'from_address']).size().unstack(fill_value=0)
        return activity_matrix

    def _create_pattern_timeline(self, df: pd.DataFrame) -> Dict:
        """Create pattern detection timeline"""
        # Simplified implementation
        return {
            'timestamp': df['timestamp'].tolist(),
            'pattern_count': [1] * len(df)  # Placeholder
        }

    def _prepare_community_data(self, graph: nx.DiGraph, communities: Dict[str, int], 
                               risk_profiles: Dict[str, Dict]) -> Dict:
        """Prepare community analysis data"""
        # Calculate community statistics
        community_stats = {}
        for node, community_id in communities.items():
            if community_id not in community_stats:
                community_stats[community_id] = {
                    'nodes': [],
                    'risk_scores': [],
                    'volumes': []
                }
            community_stats[community_id]['nodes'].append(node)
            profile = risk_profiles.get(node, {})
            community_stats[community_id]['risk_scores'].append(profile.get('risk_score', 0.0))
            community_stats[community_id]['volumes'].append(profile.get('transaction_volume', 0.0))
        
        # Calculate community sizes and average risks
        sizes = {cid: len(stats['nodes']) for cid, stats in community_stats.items()}
        risks = {cid: np.mean(stats['risk_scores']) for cid, stats in community_stats.items()}
        
        return {
            'sizes': sizes,
            'risks': risks,
            'inter_connections': {'x': [], 'y': []},  # Placeholder
            'activity_timeline': {'timestamp': [], 'activity': []}  # Placeholder
        }

    def export_visualization(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """Export visualization to file"""
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        elif format == 'json':
            fig.write_json(filename)
        
        logger.info(f"Visualization exported to {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    config = VisualizationConfig(
        mode=VisualizationMode.NETWORK_3D,
        show_risk_colors=True,
        show_communities=True
    )
    visualizer = EnhancedNetworkVisualizer(config)
    
    # Create sample data
    graph = nx.DiGraph()
    graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
    
    risk_profiles = {
        'A': {'risk_score': 0.8, 'risk_level': 'HIGH', 'transaction_count': 10, 'transaction_volume': 1.5},
        'B': {'risk_score': 0.6, 'risk_level': 'MEDIUM', 'transaction_count': 5, 'transaction_volume': 0.8},
        'C': {'risk_score': 0.9, 'risk_level': 'CRITICAL', 'transaction_count': 15, 'transaction_volume': 2.1},
        'D': {'risk_score': 0.3, 'risk_level': 'LOW', 'transaction_count': 3, 'transaction_volume': 0.4}
    }
    
    communities = {'A': 0, 'B': 0, 'C': 1, 'D': 1}
    
    # Create visualizations
    fig_3d = visualizer.create_interactive_3d_network(graph, risk_profiles, communities)
    fig_risk = visualizer.create_risk_heatmap(risk_profiles)
    
    # Export visualizations
    visualizer.export_visualization(fig_3d, 'network_3d.html')
    visualizer.export_visualization(fig_risk, 'risk_heatmap.html')
    
    print("Enhanced network visualizations created and exported")
