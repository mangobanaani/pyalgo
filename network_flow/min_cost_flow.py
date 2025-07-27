"""
Minimum Cost Flow algorithms.

This module implements algorithms for finding minimum cost flows in networks,
including the successive shortest path algorithm and cycle canceling.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import heapq
from .flow_network import FlowNetwork, Edge


class MinCostFlowNetwork(FlowNetwork):
    """
    Network flow structure with costs for minimum cost flow problems.
    
    Extends FlowNetwork to include edge costs for minimum cost flow algorithms.
    """
    
    def __init__(self, num_vertices: int):
        """Initialize minimum cost flow network."""
        super().__init__(num_vertices)
        self.costs: Dict[Tuple[int, int], int] = {}
    
    def add_edge(self, u: int, v: int, capacity: int, cost: int = 0) -> None:
        """
        Add edge with capacity and cost.
        
        Args:
            u: Source vertex
            v: Destination vertex
            capacity: Edge capacity
            cost: Cost per unit of flow
        """
        super().add_edge(u, v, capacity)
        self.costs[(u, v)] = cost
        if (v, u) not in self.costs:
            self.costs[(v, u)] = -cost  # Reverse edge has negative cost
    
    def get_cost(self, u: int, v: int) -> int:
        """Get cost of edge (u, v)."""
        return self.costs.get((u, v), 0)


def min_cost_max_flow(network: MinCostFlowNetwork, source: int, sink: int) -> Tuple[int, int]:
    """
    Find minimum cost maximum flow using successive shortest paths.
    
    Args:
        network: Network with capacities and costs
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Tuple of (max_flow_value, min_cost)
        
    Time Complexity: O(V²E²) in worst case
    Space Complexity: O(V + E)
    """
    total_flow = 0
    total_cost = 0
    
    while True:
        # Find shortest path from source to sink considering costs
        path, path_cost = _shortest_path_dijkstra(network, source, sink)
        
        if not path:
            break
        
        # Find bottleneck capacity along the path
        path_flow = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = network.get_edge(u, v)
            if edge:
                path_flow = min(path_flow, edge.capacity - edge.flow)
        
        if path_flow == 0:
            break
        
        # Augment flow along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            network.add_flow(u, v, path_flow)
        
        total_flow += path_flow
        total_cost += path_flow * path_cost
    
    return total_flow, total_cost


def min_cost_flow(network: MinCostFlowNetwork, source: int, sink: int, 
                  demand: int) -> Tuple[int, int]:
    """
    Find minimum cost flow for a specific demand.
    
    Args:
        network: Network with capacities and costs
        source: Source vertex
        sink: Sink vertex
        demand: Required flow amount
        
    Returns:
        Tuple of (achieved_flow, cost) or (-1, -1) if impossible
        
    Time Complexity: O(V²E²) in worst case
    Space Complexity: O(V + E)
    """
    total_flow = 0
    total_cost = 0
    
    while total_flow < demand:
        # Find shortest path from source to sink
        path, path_cost = _shortest_path_dijkstra(network, source, sink)
        
        if not path:
            return -1, -1  # No feasible flow
        
        # Find bottleneck capacity along the path
        path_flow = demand - total_flow
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = network.get_edge(u, v)
            if edge:
                path_flow = min(path_flow, edge.capacity - edge.flow)
        
        if path_flow == 0:
            return -1, -1  # No more capacity
        
        # Augment flow along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            network.add_flow(u, v, path_flow)
        
        total_flow += path_flow
        total_cost += path_flow * path_cost
    
    return total_flow, total_cost


def _shortest_path_dijkstra(network: MinCostFlowNetwork, source: int, 
                           sink: int) -> Tuple[List[int], int]:
    """
    Find shortest path considering costs using Dijkstra's algorithm.
    
    Only considers edges with available capacity.
    
    Returns:
        Tuple of (path, cost) or ([], 0) if no path exists
    """
    dist = [float('inf')] * network.num_vertices
    parent = [-1] * network.num_vertices
    dist[source] = 0
    
    # Priority queue: (distance, vertex)
    pq = [(0, source)]
    visited = set()
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        
        if u == sink:
            break
        
        # Check all adjacent vertices
        for v in network.graph[u]:
            if v not in visited:
                edge = network.get_edge(u, v)
                if edge and edge.capacity > edge.flow:
                    cost = network.get_cost(u, v)
                    new_dist = current_dist + cost
                    
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        parent[v] = u
                        heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct path
    if dist[sink] == float('inf'):
        return [], 0
    
    path = []
    current = sink
    while current != -1:
        path.append(current)
        current = parent[current]
    
    path.reverse()
    return path, int(dist[sink])


def cycle_canceling(network: MinCostFlowNetwork, source: int, sink: int) -> Tuple[int, int]:
    """
    Find minimum cost flow using cycle canceling algorithm.
    
    First finds any maximum flow, then cancels negative cost cycles
    to reduce cost while maintaining the same flow value.
    
    Args:
        network: Network with capacities and costs
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Tuple of (max_flow_value, min_cost)
        
    Time Complexity: O(V³E²) in worst case
    Space Complexity: O(V + E)
    """
    # First, find maximum flow using any algorithm
    from .max_flow import ford_fulkerson
    max_flow_value = ford_fulkerson(network, source, sink)
    
    # Calculate initial cost
    total_cost = _calculate_flow_cost(network)
    
    # Repeatedly find and cancel negative cost cycles
    while True:
        cycle = _find_negative_cycle(network)
        if not cycle:
            break
        
        # Find minimum residual capacity along the cycle
        cycle_flow = float('inf')
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = network.get_edge(u, v)
            if edge:
                cycle_flow = min(cycle_flow, edge.capacity - edge.flow)
        
        # Augment flow along the cycle
        if cycle_flow > 0:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                network.add_flow(u, v, cycle_flow)
    
    total_cost = _calculate_flow_cost(network)
    return max_flow_value, total_cost


def _find_negative_cycle(network: MinCostFlowNetwork) -> List[int]:
    """
    Find negative cost cycle in residual graph using Bellman-Ford.
    
    Returns:
        List of vertices forming negative cycle, or empty list if none exists
    """
    num_vertices = network.num_vertices
    dist = [0] * num_vertices
    parent = [-1] * num_vertices
    
    # Relax edges V-1 times
    for _ in range(num_vertices - 1):
        for u in range(num_vertices):
            for v in network.graph[u]:
                edge = network.get_edge(u, v)
                if edge and edge.capacity > edge.flow:
                    cost = network.get_cost(u, v)
                    if dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        parent[v] = u
    
    # Check for negative cycles
    for u in range(num_vertices):
        for v in network.graph[u]:
            edge = network.get_edge(u, v)
            if edge and edge.capacity > edge.flow:
                cost = network.get_cost(u, v)
                if dist[u] + cost < dist[v]:
                    # Found negative cycle, reconstruct it
                    cycle = []
                    visited = set()
                    current = v
                    
                    # Find a vertex in the cycle
                    for _ in range(num_vertices):
                        current = parent[current]
                    
                    # Extract the cycle
                    start = current
                    cycle.append(current)
                    current = parent[current]
                    
                    while current != start:
                        cycle.append(current)
                        current = parent[current]
                    
                    cycle.reverse()
                    return cycle
    
    return []


def _calculate_flow_cost(network: MinCostFlowNetwork) -> int:
    """Calculate total cost of current flow."""
    total_cost = 0
    
    for u in range(network.num_vertices):
        for v in network.graph[u]:
            edge = network.get_edge(u, v)
            if edge and edge.flow > 0:
                cost = network.get_cost(u, v)
                total_cost += edge.flow * cost
    
    return total_cost


def transportation_problem(supply: List[int], demand: List[int], 
                          costs: List[List[int]]) -> Tuple[int, List[List[int]]]:
    """
    Solve transportation problem using minimum cost flow.
    
    Args:
        supply: Supply at each source
        demand: Demand at each destination
        costs: Cost matrix costs[i][j] from source i to destination j
        
    Returns:
        Tuple of (minimum_cost, flow_matrix)
        
    Raises:
        ValueError: If supply and demand don't balance
    """
    if sum(supply) != sum(demand):
        raise ValueError("Total supply must equal total demand")
    
    num_sources = len(supply)
    num_destinations = len(demand)
    
    # Create network: source -> suppliers -> destinations -> sink
    num_vertices = 2 + num_sources + num_destinations
    network = MinCostFlowNetwork(num_vertices)
    
    source = 0
    sink = num_vertices - 1
    
    # Add edges from source to suppliers
    for i in range(num_sources):
        supplier = 1 + i
        network.add_edge(source, supplier, supply[i], 0)
    
    # Add edges from suppliers to destinations
    for i in range(num_sources):
        supplier = 1 + i
        for j in range(num_destinations):
            destination = 1 + num_sources + j
            network.add_edge(supplier, destination, min(supply[i], demand[j]), costs[i][j])
    
    # Add edges from destinations to sink
    for j in range(num_destinations):
        destination = 1 + num_sources + j
        network.add_edge(destination, sink, demand[j], 0)
    
    # Solve minimum cost flow
    total_demand = sum(demand)
    flow_value, min_cost = min_cost_flow(network, source, sink, total_demand)
    
    if flow_value != total_demand:
        raise ValueError("No feasible solution")
    
    # Extract flow matrix
    flow_matrix = [[0] * num_destinations for _ in range(num_sources)]
    for i in range(num_sources):
        supplier = 1 + i
        for j in range(num_destinations):
            destination = 1 + num_sources + j
            edge = network.get_edge(supplier, destination)
            if edge:
                flow_matrix[i][j] = edge.flow
    
    return min_cost, flow_matrix
