"""
Graph visualizer for document relationships and clause connections.
"""

import json
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from ..diff_engine import DiffResult, ComparisonResult
from ..semantic_comparison import SemanticDiff


@dataclass 
class GraphNode:
    """Represents a node in the relationship graph."""
    node_id: str
    label: str
    type: str  # 'clause', 'concept', 'entity', 'change'
    content: str
    position: Tuple[int, int]
    metadata: Dict[str, Any] = None


@dataclass
class GraphEdge:
    """Represents an edge in the relationship graph."""
    source_id: str
    target_id: str
    edge_type: str  # 'references', 'modifies', 'depends_on', 'replaces'
    weight: float
    label: str = ""
    metadata: Dict[str, Any] = None


class RelationshipGraphVisualizer:
    """
    Creates graph visualizations of document relationships and clause connections.
    Shows how changes affect related parts of the document.
    """
    
    def __init__(self):
        self.node_counter = 0
        self.edge_counter = 0
    
    def create_change_relationship_graph(self, old_content: str, new_content: str,
                                       comparison_result: ComparisonResult,
                                       semantic_diffs: List[SemanticDiff] = None) -> Dict[str, Any]:
        """
        Create a graph showing relationships between changes.
        
        Args:
            old_content: Original document content
            new_content: Modified document content
            comparison_result: Comparison results
            semantic_diffs: Optional semantic analysis
            
        Returns:
            Graph data and visualization
        """
        nodes = []
        edges = []
        
        # Extract clauses/sections from both documents
        old_clauses = self._extract_clauses(old_content)
        new_clauses = self._extract_clauses(new_content)
        
        # Create nodes for clauses
        clause_nodes = {}
        for clause in new_clauses:
            node = self._create_clause_node(clause)
            nodes.append(node)
            clause_nodes[clause['id']] = node
        
        # Create nodes for changes
        change_nodes = {}
        for diff in comparison_result.differences:
            node = self._create_change_node(diff)
            nodes.append(node)
            change_nodes[node.node_id] = node
            
            # Link changes to affected clauses
            affected_clauses = self._find_affected_clauses(diff, new_clauses)
            for clause in affected_clauses:
                if clause['id'] in clause_nodes:
                    edge = self._create_edge(
                        node.node_id, clause_nodes[clause['id']].node_id,
                        'affects', 1.0, f"Change affects {clause['title']}"
                    )
                    edges.append(edge)
        
        # Add semantic relationships if available
        if semantic_diffs:
            for semantic_diff in semantic_diffs:
                # Create semantic change nodes
                semantic_node = self._create_semantic_node(semantic_diff)
                nodes.append(semantic_node)
                
                # Link to related changes
                related_changes = self._find_related_changes(semantic_diff, comparison_result.differences)
                for related_change in related_changes:
                    change_node_id = f"change_{related_change.old_position[0]}"
                    if change_node_id in [n.node_id for n in nodes]:
                        edge = self._create_edge(
                            semantic_node.node_id, change_node_id,
                            'semantic_relation', semantic_diff.semantic_similarity,
                            f"Semantic relationship ({semantic_diff.change_type.value})"
                        )
                        edges.append(edge)
        
        # Find cross-references and dependencies
        cross_ref_edges = self._find_cross_references(new_clauses, clause_nodes)
        edges.extend(cross_ref_edges)
        
        # Generate visualizations
        d3_graph = self._generate_d3_graph(nodes, edges)
        cytoscape_graph = self._generate_cytoscape_graph(nodes, edges)
        
        return {
            'nodes': [self._node_to_dict(n) for n in nodes],
            'edges': [self._edge_to_dict(e) for e in edges],
            'd3_visualization': d3_graph,
            'cytoscape_visualization': cytoscape_graph,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'clause_nodes': len([n for n in nodes if n.type == 'clause']),
                'change_nodes': len([n for n in nodes if n.type == 'change']),
                'semantic_nodes': len([n for n in nodes if n.type == 'semantic_change'])
            }
        }
    
    def create_version_dependency_graph(self, versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a graph showing version dependencies and branching.
        
        Args:
            versions: List of version dictionaries
            
        Returns:
            Version dependency graph
        """
        nodes = []
        edges = []
        
        # Create nodes for each version
        version_nodes = {}
        for version in versions:
            node = GraphNode(
                node_id=version['version_id'],
                label=f"v{version['version_id'][:8]}",
                type='version',
                content=version.get('message', 'Version commit'),
                position=(0, 0),  # Will be calculated by layout algorithm
                metadata={
                    'author': version.get('author', {}),
                    'timestamp': version.get('timestamp'),
                    'status': version.get('status')
                }
            )
            nodes.append(node)
            version_nodes[version['version_id']] = node
        
        # Create edges for parent-child relationships
        for version in versions:
            if 'parent_versions' in version:
                for parent_id in version['parent_versions']:
                    if parent_id in version_nodes:
                        edge = self._create_edge(
                            parent_id, version['version_id'],
                            'parent_of', 1.0, 'Version lineage'
                        )
                        edges.append(edge)
        
        # Generate visualization
        d3_graph = self._generate_d3_version_graph(nodes, edges)
        
        return {
            'nodes': [self._node_to_dict(n) for n in nodes],
            'edges': [self._edge_to_dict(e) for e in edges],
            'd3_visualization': d3_graph,
            'statistics': {
                'total_versions': len(nodes),
                'merge_points': len([v for v in versions if len(v.get('parent_versions', [])) > 1]),
                'branch_points': len(set().union(*[v.get('parent_versions', []) for v in versions]))
            }
        }
    
    def _extract_clauses(self, content: str) -> List[Dict[str, Any]]:
        """Extract clauses or sections from document content."""
        clauses = []
        lines = content.split('\n')
        current_clause = None
        clause_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a clause header
            if self._is_clause_header(line):
                if current_clause:
                    clauses.append(current_clause)
                
                clause_counter += 1
                current_clause = {
                    'id': f"clause_{clause_counter}",
                    'title': line,
                    'content': [],
                    'start_line': i,
                    'end_line': i
                }
            elif current_clause:
                current_clause['content'].append(line)
                current_clause['end_line'] = i
        
        # Add final clause
        if current_clause:
            clauses.append(current_clause)
        
        return clauses
    
    def _is_clause_header(self, line: str) -> bool:
        """Determine if a line is a clause header."""
        import re
        
        # Common patterns for clause headers
        patterns = [
            r'^\d+\.', # Numbered clauses
            r'^[A-Z][^.]+:$', # Title case with colon
            r'^[A-Z\s]+$', # All caps
            r'^#{1,6}\s+', # Markdown headers
        ]
        
        return any(re.match(pattern, line) for pattern in patterns) and len(line) < 100
    
    def _create_clause_node(self, clause: Dict[str, Any]) -> GraphNode:
        """Create a graph node for a clause."""
        self.node_counter += 1
        
        return GraphNode(
            node_id=clause['id'],
            label=clause['title'][:50] + ('...' if len(clause['title']) > 50 else ''),
            type='clause',
            content=' '.join(clause['content'])[:200],
            position=(0, 0),
            metadata={
                'full_title': clause['title'],
                'line_range': (clause['start_line'], clause['end_line']),
                'word_count': len(' '.join(clause['content']).split())
            }
        )
    
    def _create_change_node(self, diff: DiffResult) -> GraphNode:
        """Create a graph node for a change."""
        self.node_counter += 1
        
        return GraphNode(
            node_id=f"change_{diff.old_position[0]}",
            label=f"{diff.diff_type.value.title()} Change",
            type='change',
            content=diff.new_content[:100] if diff.new_content else diff.old_content[:100],
            position=(0, 0),
            metadata={
                'diff_type': diff.diff_type.value,
                'confidence': diff.confidence,
                'old_content': diff.old_content,
                'new_content': diff.new_content
            }
        )
    
    def _create_semantic_node(self, semantic_diff: SemanticDiff) -> GraphNode:
        """Create a graph node for a semantic change."""
        self.node_counter += 1
        
        return GraphNode(
            node_id=f"semantic_{self.node_counter}",
            label=f"Semantic: {semantic_diff.change_type.value.replace('_', ' ').title()}",
            type='semantic_change',
            content=semantic_diff.explanation,
            position=(0, 0),
            metadata={
                'change_type': semantic_diff.change_type.value,
                'similarity': semantic_diff.semantic_similarity,
                'intent_change': semantic_diff.intent_change_score,
                'confidence': semantic_diff.confidence
            }
        )
    
    def _create_edge(self, source_id: str, target_id: str, edge_type: str,
                    weight: float, label: str = "") -> GraphEdge:
        """Create a graph edge."""
        self.edge_counter += 1
        
        return GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            label=label
        )
    
    def _find_affected_clauses(self, diff: DiffResult, clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find clauses affected by a change."""
        affected = []
        change_pos = diff.new_position[0] if diff.new_position else diff.old_position[0]
        
        for clause in clauses:
            if clause['start_line'] <= change_pos <= clause['end_line']:
                affected.append(clause)
        
        return affected
    
    def _find_related_changes(self, semantic_diff: SemanticDiff, 
                             differences: List[DiffResult]) -> List[DiffResult]:
        """Find changes related to a semantic change."""
        related = []
        semantic_pos = semantic_diff.position
        
        for diff in differences:
            # Check if positions overlap
            if self._positions_overlap(semantic_pos, diff.new_position):
                related.append(diff)
        
        return related
    
    def _positions_overlap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two position ranges overlap."""
        return not (pos1[1] <= pos2[0] or pos2[1] <= pos1[0])
    
    def _find_cross_references(self, clauses: List[Dict[str, Any]], 
                              clause_nodes: Dict[str, GraphNode]) -> List[GraphEdge]:
        """Find cross-references between clauses."""
        edges = []
        
        for clause in clauses:
            clause_content = ' '.join(clause['content']).lower()
            
            # Look for references to other clauses
            for other_clause in clauses:
                if clause['id'] == other_clause['id']:
                    continue
                
                # Simple reference detection - could be enhanced
                if (other_clause['title'].lower() in clause_content or
                    f"section {other_clause['id']}" in clause_content):
                    
                    edge = self._create_edge(
                        clause['id'], other_clause['id'],
                        'references', 0.5, f"References {other_clause['title'][:30]}"
                    )
                    edges.append(edge)
        
        return edges
    
    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': node.node_id,
            'label': node.label,
            'type': node.type,
            'content': node.content,
            'position': node.position,
            'metadata': node.metadata or {}
        }
    
    def _edge_to_dict(self, edge: GraphEdge) -> Dict[str, Any]:
        """Convert edge to dictionary."""
        return {
            'source': edge.source_id,
            'target': edge.target_id,
            'type': edge.edge_type,
            'weight': edge.weight,
            'label': edge.label,
            'metadata': edge.metadata or {}
        }
    
    def _generate_d3_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        """Generate D3.js graph visualization."""
        nodes_json = json.dumps([self._node_to_dict(n) for n in nodes])
        edges_json = json.dumps([self._edge_to_dict(e) for e in edges])
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Relationship Graph</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                }}
                
                .graph-container {{
                    width: 100vw;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                }}
                
                .graph-header {{
                    background: #2c3e50;
                    color: white;
                    padding: 15px 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .graph-controls {{
                    display: flex;
                    gap: 10px;
                }}
                
                .graph-controls button {{
                    background: #3498db;
                    color: white;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                
                #graph {{
                    flex: 1;
                    background: #f8f9fa;
                }}
                
                .node {{
                    cursor: pointer;
                    stroke-width: 2;
                }}
                
                .node.clause {{
                    fill: #3498db;
                    stroke: #2980b9;
                }}
                
                .node.change {{
                    fill: #e74c3c;
                    stroke: #c0392b;
                }}
                
                .node.semantic_change {{
                    fill: #9b59b6;
                    stroke: #8e44ad;
                }}
                
                .link {{
                    stroke: #95a5a6;
                    stroke-width: 1;
                    opacity: 0.6;
                }}
                
                .link.references {{
                    stroke-dasharray: 3,3;
                }}
                
                .link.affects {{
                    stroke: #e74c3c;
                    stroke-width: 2;
                }}
                
                .node-label {{
                    font-size: 10px;
                    text-anchor: middle;
                    fill: white;
                    pointer-events: none;
                }}
                
                .tooltip {{
                    position: absolute;
                    padding: 8px;
                    background: rgba(0,0,0,0.8);
                    color: white;
                    border-radius: 4px;
                    font-size: 12px;
                    pointer-events: none;
                    opacity: 0;
                    transition: opacity 0.2s;
                }}
            </style>
        </head>
        <body>
            <div class="graph-container">
                <div class="graph-header">
                    <h1>Document Relationship Graph</h1>
                    <div class="graph-controls">
                        <button onclick="restartSimulation()">Reset Layout</button>
                        <button onclick="toggleNodeLabels()">Toggle Labels</button>
                        <button onclick="filterByType('all')">All</button>
                        <button onclick="filterByType('clause')">Clauses</button>
                        <button onclick="filterByType('change')">Changes</button>
                    </div>
                </div>
                <div id="graph"></div>
            </div>
            
            <div class="tooltip" id="tooltip"></div>
            
            <script>
                const nodes = {nodes_json};
                const links = {edges_json};
                
                const width = window.innerWidth;
                const height = window.innerHeight - 60;
                
                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                const simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2));
                
                const link = svg.append("g")
                    .selectAll("line")
                    .data(links)
                    .join("line")
                    .attr("class", d => "link " + d.type)
                    .attr("stroke-width", d => Math.sqrt(d.weight * 3));
                
                const node = svg.append("g")
                    .selectAll("circle")
                    .data(nodes)
                    .join("circle")
                    .attr("class", d => "node " + d.type)
                    .attr("r", d => d.type === 'clause' ? 12 : 8)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended))
                    .on("mouseover", showTooltip)
                    .on("mouseout", hideTooltip);
                
                const label = svg.append("g")
                    .selectAll("text")
                    .data(nodes)
                    .join("text")
                    .attr("class", "node-label")
                    .text(d => d.label)
                    .style("font-size", "10px");
                
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node
                        .attr("cx", d => d.x)
                        .attr("cy", d => d.y);
                    
                    label
                        .attr("x", d => d.x)
                        .attr("y", d => d.y + 4);
                }});
                
                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
                
                function showTooltip(event, d) {{
                    const tooltip = d3.select("#tooltip");
                    tooltip.style("opacity", 1)
                        .html(d.content)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }}
                
                function hideTooltip() {{
                    d3.select("#tooltip").style("opacity", 0);
                }}
                
                function restartSimulation() {{
                    simulation.alpha(1).restart();
                }}
                
                function toggleNodeLabels() {{
                    const labels = svg.selectAll(".node-label");
                    const isVisible = labels.style("opacity") !== "0";
                    labels.style("opacity", isVisible ? 0 : 1);
                }}
                
                function filterByType(type) {{
                    node.style("opacity", d => type === 'all' || d.type === type ? 1 : 0.2);
                    label.style("opacity", d => type === 'all' || d.type === type ? 1 : 0.2);
                    link.style("opacity", d => {{
                        if (type === 'all') return 0.6;
                        const sourceVisible = nodes.find(n => n.id === d.source.id)?.type === type;
                        const targetVisible = nodes.find(n => n.id === d.target.id)?.type === type;
                        return (sourceVisible || targetVisible) ? 0.6 : 0.1;
                    }});
                }}
            </script>
        </body>
        </html>
        """
    
    def _generate_cytoscape_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        """Generate Cytoscape.js graph visualization."""
        # This would create a Cytoscape.js visualization
        # For brevity, returning a simplified version
        return json.dumps({
            'nodes': [{'data': self._node_to_dict(n)} for n in nodes],
            'edges': [{'data': self._edge_to_dict(e)} for e in edges]
        })
    
    def _generate_d3_version_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> str:
        """Generate D3.js version dependency graph."""
        # Similar to _generate_d3_graph but optimized for version trees
        return self._generate_d3_graph(nodes, edges)


# Utility function
def create_relationship_graph(old_content: str, new_content: str,
                            comparison_result: ComparisonResult) -> str:
    """
    Quick utility to create a relationship graph visualization.
    
    Args:
        old_content: Original document
        new_content: Modified document
        comparison_result: Comparison results
        
    Returns:
        HTML string with graph visualization
    """
    visualizer = RelationshipGraphVisualizer()
    result = visualizer.create_change_relationship_graph(
        old_content, new_content, comparison_result
    )
    return result['d3_visualization']