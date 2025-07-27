"""
Union-Find (Disjoint Set Union) Data Structure

Efficiently tracks a set of elements partitioned into disjoint subsets.
Supports union and find operations with path compression and union by rank optimizations.
"""

from typing import Dict, List, Optional


class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.
    
    Provides near-constant time operations for union and find operations
    through optimizations. Useful for connectivity problems and graph algorithms.
    """
    
    def __init__(self, n: Optional[int] = None):
        """
        Initialize Union-Find structure.
        
        Args:
            n: Optional initial capacity. If provided, creates sets for elements 0 to n-1.
        """
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
        self.size_map: Dict[int, int] = {}
        self._num_components = 0
        
        if n is not None:
            for i in range(n):
                self.make_set(i)
    
    def make_set(self, x: int) -> None:
        """
        Create a new set containing only element x.
        
        Time Complexity: O(1)
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size_map[x] = 1
            self._num_components += 1
    
    def find(self, x: int) -> int:
        """
        Find the representative (root) of the set containing x.
        Uses path compression for optimization.
        
        Time Complexity: O(α(n)) amortized, where α is inverse Ackermann function
        
        Args:
            x: Element to find root for
            
        Returns:
            Root element of set containing x
        """
        if x not in self.parent:
            self.make_set(x)
            return x
        
        # Path compression: make every node point directly to root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing x and y.
        Uses union by rank for optimization.
        
        Time Complexity: O(α(n)) amortized
        
        Args:
            x, y: Elements whose sets should be merged
            
        Returns:
            True if sets were merged, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        # Already in same set
        if root_x == root_y:
            return False
        
        # Union by rank: attach smaller tree under root of larger tree
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        # Make root_x the new root
        self.parent[root_y] = root_x
        self.size_map[root_x] += self.size_map[root_y]
        del self.size_map[root_y]
        
        # Update rank only if ranks were equal
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self._num_components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same set.
        
        Time Complexity: O(α(n)) amortized
        """
        return self.find(x) == self.find(y)
    
    def component_size(self, x: int) -> int:
        """
        Get size of the component containing x.
        
        Time Complexity: O(α(n)) amortized
        """
        root = self.find(x)
        return self.size_map[root]
    
    def num_components(self) -> int:
        """
        Get total number of disjoint components.
        
        Time Complexity: O(1)
        """
        return self._num_components
    
    def get_components(self) -> Dict[int, List[int]]:
        """
        Get all components as a dictionary mapping root to list of elements.
        
        Time Complexity: O(n α(n))
        """
        components: Dict[int, List[int]] = {}
        
        for element in self.parent:
            root = self.find(element)
            if root not in components:
                components[root] = []
            components[root].append(element)
        
        return components
    
    def get_all_elements(self) -> List[int]:
        """Get list of all elements in the structure."""
        return list(self.parent.keys())
    
    def reset(self) -> None:
        """Clear all sets and start fresh."""
        self.parent.clear()
        self.rank.clear()
        self.size_map.clear()
        self._num_components = 0
    
    def compress_paths(self) -> None:
        """
        Manually compress all paths for better subsequent performance.
        Useful after many operations to ensure maximum compression.
        """
        for element in list(self.parent.keys()):
            self.find(element)
    
    def __len__(self) -> int:
        """Return total number of elements."""
        return len(self.parent)
    
    def __contains__(self, x: int) -> bool:
        """Check if element x exists in any set."""
        return x in self.parent
    
    def __str__(self) -> str:
        """String representation showing all components."""
        components = self.get_components()
        if not components:
            return "UnionFind(empty)"
        
        component_strs = []
        for root, elements in components.items():
            if len(elements) == 1:
                component_strs.append(f"{{{elements[0]}}}")
            else:
                sorted_elements = sorted(elements)
                component_strs.append(f"{{{', '.join(map(str, sorted_elements))}}}")
        
        return f"UnionFind({', '.join(component_strs)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class WeightedUnionFind(UnionFind):
    """
    Weighted Union-Find that maintains weights on edges.
    Useful for problems requiring relative relationships between elements.
    """
    
    def __init__(self, n: Optional[int] = None):
        super().__init__(n)
        self.weight: Dict[int, int] = {}
    
    def make_set(self, x: int) -> None:
        """Create new set with element x having weight 0."""
        super().make_set(x)
        if x not in self.weight:
            self.weight[x] = 0
    
    def find(self, x: int) -> int:
        """Find root with path compression, updating weights along the path."""
        if x not in self.parent:
            self.make_set(x)
            return x
        
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            self.weight[x] += self.weight[original_parent]
        
        return self.parent[x]
    
    def union_with_weight(self, x: int, y: int, w: int) -> bool:
        """
        Union sets with constraint that weight[y] - weight[x] = w.
        
        Args:
            x, y: Elements to union
            w: Desired weight difference (weight[y] - weight[x])
            
        Returns:
            True if union successful, False if constraint conflicts with existing structure
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            # Check if constraint is consistent
            return self.weight[y] - self.weight[x] == w
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.weight[root_x] = self.weight[y] - self.weight[x] - w
            self.size_map[root_y] += self.size_map[root_x]
            del self.size_map[root_x]
        else:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.weight[x] - self.weight[y] + w
            self.size_map[root_x] += self.size_map[root_y]
            del self.size_map[root_y]
            
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        
        self._num_components -= 1
        return True
    
    def get_weight_difference(self, x: int, y: int) -> Optional[int]:
        """
        Get weight difference between x and y if they're connected.
        
        Returns:
            weight[y] - weight[x] if connected, None otherwise
        """
        if not self.connected(x, y):
            return None
        
        self.find(x)  # Ensure path compression
        self.find(y)
        return self.weight[y] - self.weight[x]
