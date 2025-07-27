"""
Maximum Matching algorithms for bipartite and general graphs.

This module implements algorithms for finding maximum matchings in graphs,
including the Hungarian algorithm for bipartite matching and Edmonds' blossom
algorithm for general graphs.
"""

from typing import List, Set, Dict, Optional, Tuple
from collections import defaultdict, deque
from .flow_network import FlowNetwork


def maximum_bipartite_matching(graph: Dict[int, List[int]], 
                              left_vertices: Set[int], 
                              right_vertices: Set[int]) -> List[Tuple[int, int]]:
    """
    Find maximum matching in bipartite graph using Ford-Fulkerson.
    
    Args:
        graph: Adjacency list representation
        left_vertices: Set of vertices in left partition
        right_vertices: Set of vertices in right partition
        
    Returns:
        List of matched pairs (u, v)
        
    Time Complexity: O(VE)
    Space Complexity: O(V + E)
    """
    # Convert to flow network
    source = max(max(left_vertices), max(right_vertices)) + 1
    sink = source + 1
    
    network = FlowNetwork(sink + 1)
    
    # Add edges from source to left vertices
    for u in left_vertices:
        network.add_edge(source, u, 1)
    
    # Add edges between left and right vertices
    for u in left_vertices:
        for v in graph.get(u, []):
            if v in right_vertices:
                network.add_edge(u, v, 1)
    
    # Add edges from right vertices to sink
    for v in right_vertices:
        network.add_edge(v, sink, 1)
    
    # Find maximum flow
    from .max_flow import ford_fulkerson
    max_flow = ford_fulkerson(network, source, sink)
    
    # Extract matching from flow
    matching = []
    for u in left_vertices:
        for v in graph.get(u, []):
            if v in right_vertices:
                edge = network.get_edge(u, v)
                if edge and edge.flow > 0:
                    matching.append((u, v))
    
    return matching


def hopcroft_karp(graph: Dict[int, List[int]], 
                  left_vertices: Set[int], 
                  right_vertices: Set[int]) -> List[Tuple[int, int]]:
    """
    Find maximum bipartite matching using Hopcroft-Karp algorithm.
    
    More efficient than Ford-Fulkerson for bipartite matching.
    
    Args:
        graph: Adjacency list representation
        left_vertices: Set of vertices in left partition
        right_vertices: Set of vertices in right partition
        
    Returns:
        List of matched pairs (u, v)
        
    Time Complexity: O(E√V)
    Space Complexity: O(V)
    """
    # Initialize matching
    match_left = {u: None for u in left_vertices}
    match_right = {v: None for v in right_vertices}
    
    matching = 0
    
    while True:
        # Build layer graph using BFS
        if not _bfs_hopcroft_karp(graph, left_vertices, right_vertices, 
                                 match_left, match_right):
            break
        
        # Find augmenting paths using DFS
        for u in left_vertices:
            if match_left[u] is None:
                if _dfs_hopcroft_karp(graph, u, left_vertices, right_vertices,
                                     match_left, match_right, set()):
                    matching += 1
    
    # Extract matching pairs
    result = []
    for u in left_vertices:
        if match_left[u] is not None:
            result.append((u, match_left[u]))
    
    return result


def _bfs_hopcroft_karp(graph: Dict[int, List[int]], 
                      left_vertices: Set[int], right_vertices: Set[int],
                      match_left: Dict[int, Optional[int]], 
                      match_right: Dict[int, Optional[int]]) -> bool:
    """BFS to build layer graph for Hopcroft-Karp."""
    # Distance from unmatched left vertices
    dist = {}
    queue = deque()
    
    # Initialize distances
    for u in left_vertices:
        if match_left[u] is None:
            dist[u] = 0
            queue.append(u)
        else:
            dist[u] = float('inf')
    
    dist[None] = float('inf')
    
    while queue:
        u = queue.popleft()
        
        if dist[u] < dist[None]:
            for v in graph.get(u, []):
                if v in right_vertices:
                    if dist.get(match_right[v], float('inf')) == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
    
    return dist[None] != float('inf')


def _dfs_hopcroft_karp(graph: Dict[int, List[int]], u: int,
                      left_vertices: Set[int], right_vertices: Set[int],
                      match_left: Dict[int, Optional[int]], 
                      match_right: Dict[int, Optional[int]],
                      visited: Set[int]) -> bool:
    """DFS to find augmenting paths for Hopcroft-Karp."""
    if u in visited:
        return False
    
    visited.add(u)
    
    for v in graph.get(u, []):
        if v in right_vertices:
            if (match_right[v] is None or 
                _dfs_hopcroft_karp(graph, match_right[v], left_vertices, right_vertices,
                                  match_left, match_right, visited)):
                match_left[u] = v
                match_right[v] = u
                return True
    
    return False


def hungarian_algorithm(cost_matrix: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Solve assignment problem using Hungarian algorithm.
    
    Finds minimum cost perfect matching in complete bipartite graph.
    
    Args:
        cost_matrix: Cost matrix where cost_matrix[i][j] is cost of assigning
                    worker i to job j
        
    Returns:
        Tuple of (minimum_cost, assignment_pairs)
        
    Time Complexity: O(n³)
    Space Complexity: O(n²)
    """
    n = len(cost_matrix)
    if n == 0 or any(len(row) != n for row in cost_matrix):
        raise ValueError("Cost matrix must be square and non-empty")
    
    # Create augmented cost matrix
    cost = [[cost_matrix[i][j] for j in range(n)] for i in range(n)]
    
    # Step 1: Subtract row minima
    for i in range(n):
        row_min = min(cost[i])
        for j in range(n):
            cost[i][j] -= row_min
    
    # Step 2: Subtract column minima
    for j in range(n):
        col_min = min(cost[i][j] for i in range(n))
        for i in range(n):
            cost[i][j] -= col_min
    
    # Initialize matching
    row_match = [-1] * n
    col_match = [-1] * n
    
    # Try to find initial matching
    for i in range(n):
        for j in range(n):
            if cost[i][j] == 0 and row_match[i] == -1 and col_match[j] == -1:
                row_match[i] = j
                col_match[j] = i
    
    # Augment matching until perfect
    matched_rows = sum(1 for x in row_match if x != -1)
    
    while matched_rows < n:
        # Find minimum vertex cover
        marked_rows = set()
        marked_cols = set()
        
        # Mark unmatched rows
        for i in range(n):
            if row_match[i] == -1:
                marked_rows.add(i)
        
        # Iteratively mark rows and columns
        changed = True
        while changed:
            changed = False
            
            # Mark columns with zeros in marked rows
            for i in marked_rows:
                for j in range(n):
                    if cost[i][j] == 0 and j not in marked_cols:
                        marked_cols.add(j)
                        changed = True
            
            # Mark rows matched to marked columns
            for j in marked_cols:
                if col_match[j] != -1 and col_match[j] not in marked_rows:
                    marked_rows.add(col_match[j])
                    changed = True
        
        # Find minimum value in uncovered elements
        min_val = float('inf')
        for i in range(n):
            if i not in marked_rows:
                for j in range(n):
                    if j not in marked_cols:
                        min_val = min(min_val, cost[i][j])
        
        # Update cost matrix
        for i in range(n):
            for j in range(n):
                if i not in marked_rows and j not in marked_cols:
                    cost[i][j] -= min_val
                elif i in marked_rows and j in marked_cols:
                    cost[i][j] += min_val
        
        # Find new matching
        row_match = [-1] * n
        col_match = [-1] * n
        
        for i in range(n):
            for j in range(n):
                if cost[i][j] == 0 and row_match[i] == -1 and col_match[j] == -1:
                    row_match[i] = j
                    col_match[j] = i
        
        matched_rows = sum(1 for x in row_match if x != -1)
    
    # Calculate total cost and create assignment
    total_cost = 0
    assignment = []
    for i in range(n):
        j = row_match[i]
        total_cost += cost_matrix[i][j]
        assignment.append((i, j))
    
    return total_cost, assignment


def maximum_weight_matching(graph: Dict[int, Dict[int, int]]) -> List[Tuple[int, int]]:
    """
    Find maximum weight matching in general graph.
    
    Simplified implementation for small graphs.
    
    Args:
        graph: Adjacency dictionary with weights graph[u][v] = weight
        
    Returns:
        List of matched pairs with maximum total weight
        
    Time Complexity: O(V!) - exponential (brute force for simplicity)
    Space Complexity: O(V)
    """
    vertices = list(graph.keys())
    n = len(vertices)
    
    if n == 0:
        return []
    
    best_matching = []
    best_weight = 0
    
    # Try all possible matchings (brute force)
    def backtrack(used: Set[int], current_matching: List[Tuple[int, int]], 
                  current_weight: int) -> None:
        nonlocal best_matching, best_weight
        
        if current_weight > best_weight:
            best_weight = current_weight
            best_matching = current_matching.copy()
        
        # Try to add more edges to matching
        for u in vertices:
            if u in used:
                continue
            
            for v in graph.get(u, {}):
                if v in used or v <= u:  # Avoid duplicates and self-loops
                    continue
                
                # Add edge (u, v) to matching
                used.add(u)
                used.add(v)
                current_matching.append((u, v))
                
                backtrack(used, current_matching, current_weight + graph[u][v])
                
                # Backtrack
                used.remove(u)
                used.remove(v)
                current_matching.pop()
    
    backtrack(set(), [], 0)
    return best_matching


def stable_marriage(men_prefs: Dict[int, List[int]], 
                   women_prefs: Dict[int, List[int]]) -> Dict[int, int]:
    """
    Solve stable marriage problem using Gale-Shapley algorithm.
    
    Args:
        men_prefs: Preference lists for men {man_id: [woman_id, ...]}
        women_prefs: Preference lists for women {woman_id: [man_id, ...]}
        
    Returns:
        Matching dictionary {man_id: woman_id}
        
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    # Initialize
    men = set(men_prefs.keys())
    women = set(women_prefs.keys())
    
    if len(men) != len(women):
        raise ValueError("Number of men and women must be equal")
    
    # Current proposals for each man
    next_proposal = {man: 0 for man in men}
    
    # Current partner for each woman (None if unmatched)
    woman_partner = {woman: None for woman in women}
    
    # Preference ranking for women (for quick comparison)
    women_ranking = {}
    for woman, prefs in women_prefs.items():
        women_ranking[woman] = {man: i for i, man in enumerate(prefs)}
    
    free_men = list(men)
    
    while free_men:
        man = free_men.pop(0)
        
        # Get next woman on man's preference list
        if next_proposal[man] >= len(men_prefs[man]):
            continue  # No more women to propose to
        
        woman = men_prefs[man][next_proposal[man]]
        next_proposal[man] += 1
        
        if woman_partner[woman] is None:
            # Woman is free, engage them
            woman_partner[woman] = man
        else:
            # Woman is already engaged, check if she prefers this man
            current_partner = woman_partner[woman]
            
            if (women_ranking[woman][man] < women_ranking[woman][current_partner]):
                # Woman prefers new man
                woman_partner[woman] = man
                free_men.append(current_partner)  # Current partner becomes free
            else:
                # Woman prefers current partner
                free_men.append(man)  # Man remains free
    
    # Create matching from woman's perspective to man's perspective
    matching = {}
    for woman, man in woman_partner.items():
        if man is not None:
            matching[man] = woman
    
    return matching
