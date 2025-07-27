from collections import defaultdict, deque


class Graph:
    """
    Graph class with adjacency list representation
    """
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed
    
    def add_edge(self, u, v, weight=1):
        """Add an edge between vertices u and v"""
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def add_vertex(self, vertex):
        """Add a vertex to the graph"""
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def get_vertices(self):
        """Get all vertices in the graph"""
        vertices = set()
        for vertex in self.graph:
            vertices.add(vertex)
            for neighbor, _ in self.graph[vertex]:
                vertices.add(neighbor)
        return list(vertices)
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    def display(self):
        """Display the graph"""
        for vertex in self.graph:
            print(f"{vertex}: {[neighbor for neighbor, weight in self.graph[vertex]]}")


class GraphAlgorithms:
    """
    Collection of graph algorithms
    """
    
    @staticmethod
    def bfs(graph, start):
        """
        Breadth-First Search (BFS) traversal
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set()
        queue = deque([start])
        visited.add(start)
        traversal = []
        
        while queue:
            vertex = queue.popleft()
            traversal.append(vertex)
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return traversal
    
    @staticmethod
    def dfs(graph, start, visited=None):
        """
        Depth-First Search (DFS) traversal (recursive)
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        if visited is None:
            visited = set()
        
        visited.add(start)
        traversal = [start]
        
        for neighbor, _ in graph.get_neighbors(start):
            if neighbor not in visited:
                traversal.extend(GraphAlgorithms.dfs(graph, neighbor, visited))
        
        return traversal
    
    @staticmethod
    def dfs_iterative(graph, start):
        """
        Depth-First Search (DFS) traversal (iterative)
        """
        visited = set()
        stack = [start]
        traversal = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                traversal.append(vertex)
                
                # Add neighbors to stack (reverse order for left-to-right traversal)
                neighbors = [neighbor for neighbor, _ in graph.get_neighbors(vertex)]
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return traversal
    
    @staticmethod
    def has_path(graph, start, end):
        """
        Check if there's a path between start and end vertices using BFS
        """
        if start == end:
            return True
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            vertex = queue.popleft()
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor == end:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    @staticmethod
    def find_shortest_path(graph, start, end):
        """
        Find shortest path between start and end using BFS (unweighted graph)
        Returns the path as a list of vertices
        """
        if start == end:
            return [start]
        
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        
        while queue:
            vertex, path = queue.popleft()
            
            for neighbor, _ in graph.get_neighbors(vertex):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
