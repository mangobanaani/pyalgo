"""
Flow Network Data Structure

Basic data structure for representing flow networks.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Edge:
    """Represents an edge in a flow network."""
    to: int
    capacity: int
    flow: int = 0
    cost: int = 0  # For min-cost flow problems
    
    def residual_capacity(self) -> int:
        """Return the residual capacity of this edge."""
        return self.capacity - self.flow
    
    def is_saturated(self) -> bool:
        """Check if this edge is saturated (flow equals capacity)."""
        return self.flow >= self.capacity

class FlowNetwork:
    """
    Flow network representation using adjacency list.
    """
    
    def __init__(self, n: int):
        """
        Initialize flow network with n vertices.
        
        Args:
            n: Number of vertices
        """
        self.n = n
        self.graph: List[List[Edge]] = [[] for _ in range(n)]
        self.edges: List[Edge] = []  # All edges for easy access
    
    def add_edge(self, from_vertex: int, to_vertex: int, capacity: int, cost: int = 0) -> int:
        """
        Add an edge to the flow network.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            capacity: Edge capacity
            cost: Edge cost (for min-cost flow)
            
        Returns:
            Index of the added edge
        """
        # Forward edge
        forward_edge = Edge(to_vertex, capacity, 0, cost)
        # Backward edge (residual edge with 0 capacity initially)
        backward_edge = Edge(from_vertex, 0, 0, -cost)
        
        # Add edges to adjacency list
        self.graph[from_vertex].append(forward_edge)
        self.graph[to_vertex].append(backward_edge)
        
        # Store edges for easy access
        edge_index = len(self.edges)
        self.edges.append(forward_edge)
        self.edges.append(backward_edge)
        
        return edge_index
    
    def get_neighbors(self, vertex: int) -> List[Edge]:
        """Get all neighbors of a vertex."""
        return self.graph[vertex]
    
    def reset_flow(self):
        """Reset all flows to zero."""
        for edge in self.edges:
            edge.flow = 0
    
    def total_flow_from_source(self, source: int) -> int:
        """Calculate total flow leaving the source."""
        total = 0
        for edge in self.graph[source]:
            total += edge.flow
        return total
    
    def is_valid_flow(self, source: int, sink: int) -> bool:
        """
        Check if current flow satisfies flow conservation constraints.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            True if flow is valid
        """
        for vertex in range(self.n):
            if vertex == source or vertex == sink:
                continue
            
            flow_in = 0
            flow_out = 0
            
            # Calculate flow in
            for from_v in range(self.n):
                for edge in self.graph[from_v]:
                    if edge.to == vertex:
                        flow_in += edge.flow
            
            # Calculate flow out
            for edge in self.graph[vertex]:
                flow_out += edge.flow
            
            # Flow conservation: flow in = flow out
            if flow_in != flow_out:
                return False
        
        return True
    
    def get_flow_value(self, source: int, sink: int) -> int:
        """Get the value of the current flow."""
        flow_out = sum(edge.flow for edge in self.graph[source])
        flow_in = sum(edge.flow for from_v in range(self.n) 
                     for edge in self.graph[from_v] if edge.to == sink)
        return min(flow_out, flow_in)
    
    def get_total_cost(self) -> int:
        """Calculate total cost of current flow."""
        total_cost = 0
        for edge in self.edges[::2]:  # Only consider forward edges
            total_cost += edge.flow * edge.cost
        return total_cost
    
    def print_flow(self):
        """Print the current flow in the network."""
        print("Current flow:")
        for from_v in range(self.n):
            for edge in self.graph[from_v]:
                if edge.flow > 0:
                    print(f"  {from_v} -> {edge.to}: {edge.flow}/{edge.capacity}")
    
    def to_capacity_matrix(self) -> List[List[int]]:
        """Convert to capacity matrix representation."""
        matrix = [[0] * self.n for _ in range(self.n)]
        for from_v in range(self.n):
            for edge in self.graph[from_v]:
                matrix[from_v][edge.to] = edge.capacity
        return matrix
    
    def to_flow_matrix(self) -> List[List[int]]:
        """Convert to flow matrix representation."""
        matrix = [[0] * self.n for _ in range(self.n)]
        for from_v in range(self.n):
            for edge in self.graph[from_v]:
                matrix[from_v][edge.to] = edge.flow
        return matrix
    
    @classmethod
    def from_capacity_matrix(cls, capacity_matrix: List[List[int]]) -> 'FlowNetwork':
        """
        Create flow network from capacity matrix.
        
        Args:
            capacity_matrix: n x n matrix where capacity_matrix[i][j] is the capacity from i to j
            
        Returns:
            FlowNetwork instance
        """
        n = len(capacity_matrix)
        network = cls(n)
        
        for i in range(n):
            for j in range(n):
                if capacity_matrix[i][j] > 0:
                    network.add_edge(i, j, capacity_matrix[i][j])
        
        return network
    
    def add_source_sink(self, sources: List[int], sinks: List[int], 
                       source_capacities: Optional[List[int]] = None,
                       sink_capacities: Optional[List[int]] = None) -> tuple:
        """
        Add a super source and super sink to the network.
        
        Args:
            sources: List of source vertices
            sinks: List of sink vertices
            source_capacities: Capacities from super source to sources
            sink_capacities: Capacities from sinks to super sink
            
        Returns:
            Tuple of (super_source_index, super_sink_index)
        """
        super_source = self.n
        super_sink = self.n + 1
        
        # Expand the network
        self.n += 2
        self.graph.extend([[], []])
        
        # Connect super source to sources
        if source_capacities is None:
            source_capacities = [float('inf')] * len(sources)
        
        for i, source in enumerate(sources):
            self.add_edge(super_source, source, source_capacities[i])
        
        # Connect sinks to super sink
        if sink_capacities is None:
            sink_capacities = [float('inf')] * len(sinks)
        
        for i, sink in enumerate(sinks):
            self.add_edge(sink, super_sink, sink_capacities[i])
        
        return super_source, super_sink
