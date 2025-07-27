"""
Maximum Flow Algorithms

Implementations of various maximum flow algorithms.
"""

from collections import deque
from typing import List, Optional
from .flow_network import FlowNetwork, Edge

def ford_fulkerson(network: FlowNetwork, source: int, sink: int) -> int:
    """
    Ford-Fulkerson algorithm for maximum flow using DFS to find augmenting paths.
    
    Time Complexity: O(E * max_flow) where E is number of edges
    Space Complexity: O(V) where V is number of vertices
    
    Args:
        network: Flow network
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Maximum flow value
    """
    network.reset_flow()
    max_flow = 0
    
    while True:
        # Find augmenting path using DFS
        parent = [-1] * network.n
        visited = [False] * network.n
        
        if not _dfs_augmenting_path(network, source, sink, visited, parent):
            break
        
        # Find minimum residual capacity along the path
        path_flow = float('inf')
        current = sink
        
        while current != source:
            prev = parent[current]
            # Find the edge from prev to current
            for edge in network.graph[prev]:
                if edge.to == current:
                    path_flow = min(path_flow, edge.residual_capacity())
                    break
            current = prev
        
        # Update residual capacities
        current = sink
        while current != source:
            prev = parent[current]
            # Update forward edge
            for edge in network.graph[prev]:
                if edge.to == current:
                    edge.flow += path_flow
                    break
            # Update backward edge
            for edge in network.graph[current]:
                if edge.to == prev:
                    edge.flow -= path_flow
                    break
            current = prev
        
        max_flow += path_flow
    
    return max_flow

def _dfs_augmenting_path(network: FlowNetwork, current: int, sink: int, 
                        visited: List[bool], parent: List[int]) -> bool:
    """
    DFS to find augmenting path from current to sink.
    
    Args:
        network: Flow network
        current: Current vertex
        sink: Sink vertex
        visited: Visited array
        parent: Parent array for path reconstruction
        
    Returns:
        True if path to sink is found
    """
    if current == sink:
        return True
    
    visited[current] = True
    
    for edge in network.graph[current]:
        if not visited[edge.to] and edge.residual_capacity() > 0:
            parent[edge.to] = current
            if _dfs_augmenting_path(network, edge.to, sink, visited, parent):
                return True
    
    return False

def edmonds_karp(network: FlowNetwork, source: int, sink: int) -> int:
    """
    Edmonds-Karp algorithm for maximum flow using BFS to find shortest augmenting paths.
    
    Time Complexity: O(V * E²) where V is vertices and E is edges
    Space Complexity: O(V)
    
    Args:
        network: Flow network
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Maximum flow value
    """
    network.reset_flow()
    max_flow = 0
    
    while True:
        # Find shortest augmenting path using BFS
        parent = [-1] * network.n
        visited = [False] * network.n
        queue = deque([source])
        visited[source] = True
        
        # BFS to find augmenting path
        found_path = False
        while queue and not found_path:
            current = queue.popleft()
            
            for edge in network.graph[current]:
                if not visited[edge.to] and edge.residual_capacity() > 0:
                    visited[edge.to] = True
                    parent[edge.to] = current
                    queue.append(edge.to)
                    
                    if edge.to == sink:
                        found_path = True
                        break
        
        if not found_path:
            break
        
        # Find minimum residual capacity along the path
        path_flow = float('inf')
        current = sink
        
        while current != source:
            prev = parent[current]
            for edge in network.graph[prev]:
                if edge.to == current:
                    path_flow = min(path_flow, edge.residual_capacity())
                    break
            current = prev
        
        # Update residual capacities
        current = sink
        while current != source:
            prev = parent[current]
            # Update forward edge
            for edge in network.graph[prev]:
                if edge.to == current:
                    edge.flow += path_flow
                    break
            # Update backward edge
            for edge in network.graph[current]:
                if edge.to == prev:
                    edge.flow -= path_flow
                    break
            current = prev
        
        max_flow += path_flow
    
    return max_flow

def dinic_algorithm(network: FlowNetwork, source: int, sink: int) -> int:
    """
    Dinic's algorithm for maximum flow using level graphs and blocking flows.
    
    Time Complexity: O(V² * E) where V is vertices and E is edges
    Space Complexity: O(V)
    
    Args:
        network: Flow network
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Maximum flow value
    """
    network.reset_flow()
    max_flow = 0
    
    while True:
        # Build level graph using BFS
        level = [-1] * network.n
        level[source] = 0
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            
            for edge in network.graph[current]:
                if level[edge.to] < 0 and edge.residual_capacity() > 0:
                    level[edge.to] = level[current] + 1
                    queue.append(edge.to)
        
        # If sink is not reachable, no more augmenting paths
        if level[sink] < 0:
            break
        
        # Find blocking flow using DFS
        iter_count = [0] * network.n
        
        while True:
            flow = _dinic_dfs(network, source, sink, float('inf'), level, iter_count)
            if flow == 0:
                break
            max_flow += flow
    
    return max_flow

def _dinic_dfs(network: FlowNetwork, current: int, sink: int, flow: int,
               level: List[int], iter_count: List[int]) -> int:
    """
    DFS for finding blocking flow in Dinic's algorithm.
    
    Args:
        network: Flow network
        current: Current vertex
        sink: Sink vertex
        flow: Current flow amount
        level: Level array from BFS
        iter_count: Iterator count for each vertex
        
    Returns:
        Flow pushed through this path
    """
    if current == sink:
        return flow
    
    while iter_count[current] < len(network.graph[current]):
        edge = network.graph[current][iter_count[current]]
        
        if (level[edge.to] == level[current] + 1 and 
            edge.residual_capacity() > 0):
            
            min_flow = min(flow, edge.residual_capacity())
            pushed_flow = _dinic_dfs(network, edge.to, sink, min_flow, level, iter_count)
            
            if pushed_flow > 0:
                edge.flow += pushed_flow
                # Find and update backward edge
                for back_edge in network.graph[edge.to]:
                    if back_edge.to == current:
                        back_edge.flow -= pushed_flow
                        break
                return pushed_flow
        
        iter_count[current] += 1
    
    return 0

def maximum_flow_push_relabel(network: FlowNetwork, source: int, sink: int) -> int:
    """
    Push-relabel algorithm for maximum flow.
    
    Time Complexity: O(V²√E) with FIFO selection rule
    Space Complexity: O(V)
    
    Args:
        network: Flow network
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Maximum flow value
    """
    network.reset_flow()
    n = network.n
    
    # Initialize preflow
    height = [0] * n
    excess = [0] * n
    height[source] = n
    
    # Saturate all edges from source
    for edge in network.graph[source]:
        edge.flow = edge.capacity
        excess[edge.to] += edge.capacity
        excess[source] -= edge.capacity
        
        # Update backward edge
        for back_edge in network.graph[edge.to]:
            if back_edge.to == source:
                back_edge.flow = -edge.capacity
                break
    
    # Main loop
    changed = True
    while changed:
        changed = False
        
        for vertex in range(n):
            if vertex == source or vertex == sink or excess[vertex] <= 0:
                continue
            
            # Try to push flow
            for edge in network.graph[vertex]:
                if (excess[vertex] > 0 and edge.residual_capacity() > 0 and 
                    height[vertex] == height[edge.to] + 1):
                    
                    # Push flow
                    push_flow = min(excess[vertex], edge.residual_capacity())
                    edge.flow += push_flow
                    excess[vertex] -= push_flow
                    excess[edge.to] += push_flow
                    
                    # Update backward edge
                    for back_edge in network.graph[edge.to]:
                        if back_edge.to == vertex:
                            back_edge.flow -= push_flow
                            break
                    
                    changed = True
                    
                    if excess[vertex] == 0:
                        break
            
            # If still have excess, relabel
            if excess[vertex] > 0:
                min_height = float('inf')
                for edge in network.graph[vertex]:
                    if edge.residual_capacity() > 0:
                        min_height = min(min_height, height[edge.to])
                
                if min_height < float('inf'):
                    height[vertex] = min_height + 1
                    changed = True
    
    return sum(edge.flow for edge in network.graph[source])

def get_min_cut(network: FlowNetwork, source: int, sink: int) -> tuple:
    """
    Find minimum cut after computing maximum flow.
    
    Args:
        network: Flow network (should have max flow computed)
        source: Source vertex
        sink: Sink vertex
        
    Returns:
        Tuple of (cut_edges, cut_capacity, reachable_vertices)
    """
    # Find all vertices reachable from source in residual graph
    visited = [False] * network.n
    queue = deque([source])
    visited[source] = True
    reachable = {source}
    
    while queue:
        current = queue.popleft()
        
        for edge in network.graph[current]:
            if not visited[edge.to] and edge.residual_capacity() > 0:
                visited[edge.to] = True
                reachable.add(edge.to)
                queue.append(edge.to)
    
    # Find cut edges
    cut_edges = []
    cut_capacity = 0
    
    for vertex in reachable:
        for edge in network.graph[vertex]:
            if edge.to not in reachable:
                cut_edges.append((vertex, edge.to))
                cut_capacity += edge.capacity
    
    return cut_edges, cut_capacity, reachable
