"""
Persistent Segment Tree Implementation

Immutable segment tree that preserves all versions for historical queries.
Supports range queries and updates while maintaining query history.
"""

from typing import Any, Optional, Callable, List, Tuple
import copy


class PersistentSegmentNode:
    """Node in a persistent segment tree."""
    
    def __init__(self, value: Any, left: Optional['PersistentSegmentNode'] = None,
                 right: Optional['PersistentSegmentNode'] = None):
        self.value = value
        self.left = left
        self.right = right
        self.version = 0
    
    def copy(self) -> 'PersistentSegmentNode':
        """Create a copy of the node."""
        new_node = PersistentSegmentNode(self.value, self.left, self.right)
        new_node.version = self.version + 1
        return new_node


class PersistentSegmentTree:
    """
    Persistent Segment Tree implementation.
    
    Maintains all versions of the tree, allowing queries on historical states.
    Each update creates a new version while sharing unchanged nodes.
    """
    
    def __init__(self, data: List[Any], combine_func: Callable[[Any, Any], Any],
                 identity: Any):
        """
        Initialize persistent segment tree.
        
        Args:
            data: Initial array data
            combine_func: Function to combine two values (e.g., min, max, sum)
            identity: Identity element for combine_func
        """
        self.combine_func = combine_func
        self.identity = identity
        self.n = len(data)
        self.versions: List[PersistentSegmentNode] = []
        
        # Build initial tree
        if data:
            root = self._build(data, 0, self.n - 1)
            self.versions.append(root)
        else:
            self.versions.append(None)
    
    def _build(self, data: List[Any], left: int, right: int) -> PersistentSegmentNode:
        """Build segment tree from data."""
        if left == right:
            return PersistentSegmentNode(data[left])
        
        mid = (left + right) // 2
        left_child = self._build(data, left, mid)
        right_child = self._build(data, mid + 1, right)
        
        combined_value = self.combine_func(left_child.value, right_child.value)
        return PersistentSegmentNode(combined_value, left_child, right_child)
    
    def update(self, version: int, index: int, value: Any) -> int:
        """
        Update value at index in given version.
        
        Args:
            version: Version to update
            index: Index to update
            value: New value
            
        Returns:
            New version number
            
        Raises:
            IndexError: If version or index is invalid
        """
        if not (0 <= version < len(self.versions)):
            raise IndexError(f"Invalid version {version}")
        
        if not (0 <= index < self.n):
            raise IndexError(f"Invalid index {index}")
        
        old_root = self.versions[version]
        new_root = self._update_recursive(old_root, 0, self.n - 1, index, value)
        
        self.versions.append(new_root)
        return len(self.versions) - 1
    
    def _update_recursive(self, node: Optional[PersistentSegmentNode],
                         left: int, right: int, index: int, value: Any) -> PersistentSegmentNode:
        """Recursively update value and create new nodes."""
        if left == right:
            # Leaf node
            return PersistentSegmentNode(value)
        
        mid = (left + right) // 2
        new_node = node.copy() if node else PersistentSegmentNode(self.identity)
        
        if index <= mid:
            # Update left subtree
            new_node.left = self._update_recursive(new_node.left, left, mid, index, value)
        else:
            # Update right subtree
            new_node.right = self._update_recursive(new_node.right, mid + 1, right, index, value)
        
        # Update current node value
        left_value = new_node.left.value if new_node.left else self.identity
        right_value = new_node.right.value if new_node.right else self.identity
        new_node.value = self.combine_func(left_value, right_value)
        
        return new_node
    
    def query(self, version: int, query_left: int, query_right: int) -> Any:
        """
        Query range in given version.
        
        Args:
            version: Version to query
            query_left: Left bound of query (inclusive)
            query_right: Right bound of query (inclusive)
            
        Returns:
            Combined value over the range
            
        Raises:
            IndexError: If version or range is invalid
        """
        if not (0 <= version < len(self.versions)):
            raise IndexError(f"Invalid version {version}")
        
        if not (0 <= query_left <= query_right < self.n):
            raise IndexError(f"Invalid range [{query_left}, {query_right}]")
        
        root = self.versions[version]
        return self._query_recursive(root, 0, self.n - 1, query_left, query_right)
    
    def _query_recursive(self, node: Optional[PersistentSegmentNode],
                        left: int, right: int, query_left: int, query_right: int) -> Any:
        """Recursively query range."""
        if node is None or query_left > right or query_right < left:
            return self.identity
        
        if query_left <= left and right <= query_right:
            # Current range is completely within query range
            return node.value
        
        # Partial overlap
        mid = (left + right) // 2
        left_result = self._query_recursive(node.left, left, mid, query_left, query_right)
        right_result = self._query_recursive(node.right, mid + 1, right, query_left, query_right)
        
        return self.combine_func(left_result, right_result)
    
    def point_query(self, version: int, index: int) -> Any:
        """
        Query single point in given version.
        
        Args:
            version: Version to query
            index: Index to query
            
        Returns:
            Value at index
        """
        return self.query(version, index, index)
    
    def range_update(self, version: int, update_left: int, update_right: int, 
                    value: Any) -> int:
        """
        Update range with value (for additive operations).
        
        Args:
            version: Version to update
            update_left: Left bound of update (inclusive)
            update_right: Right bound of update (inclusive)
            value: Value to add/apply
            
        Returns:
            New version number
        """
        current_version = version
        
        for i in range(update_left, update_right + 1):
            old_value = self.point_query(current_version, i)
            new_value = self.combine_func(old_value, value)
            current_version = self.update(current_version, i, new_value)
        
        return current_version
    
    def get_version_count(self) -> int:
        """Return number of versions."""
        return len(self.versions)
    
    def get_latest_version(self) -> int:
        """Return latest version number."""
        return len(self.versions) - 1
    
    def get_array(self, version: int) -> List[Any]:
        """
        Get array representation of given version.
        
        Args:
            version: Version to convert
            
        Returns:
            List representation of the version
        """
        if not (0 <= version < len(self.versions)):
            raise IndexError(f"Invalid version {version}")
        
        result = []
        for i in range(self.n):
            result.append(self.point_query(version, i))
        return result
    
    def get_tree_size(self, version: int) -> int:
        """
        Get number of nodes in given version (for memory analysis).
        
        Args:
            version: Version to analyze
            
        Returns:
            Number of nodes in the tree
        """
        if not (0 <= version < len(self.versions)):
            raise IndexError(f"Invalid version {version}")
        
        return self._count_nodes(self.versions[version])
    
    def _count_nodes(self, node: Optional[PersistentSegmentNode]) -> int:
        """Count nodes in subtree."""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
    
    def print_version(self, version: int) -> None:
        """Print tree structure for debugging."""
        if not (0 <= version < len(self.versions)):
            raise IndexError(f"Invalid version {version}")
        
        print(f"Version {version}:")
        self._print_tree(self.versions[version], 0, self.n - 1, 0)
    
    def _print_tree(self, node: Optional[PersistentSegmentNode],
                   left: int, right: int, depth: int) -> None:
        """Print tree structure recursively."""
        if node is None:
            return
        
        indent = "  " * depth
        if left == right:
            print(f"{indent}Leaf[{left}]: {node.value}")
        else:
            print(f"{indent}Node[{left}, {right}]: {node.value}")
            mid = (left + right) // 2
            self._print_tree(node.left, left, mid, depth + 1)
            self._print_tree(node.right, mid + 1, right, depth + 1)
    
    def compress_versions(self, keep_versions: List[int]) -> None:
        """
        Compress versions by keeping only specified versions.
        
        Args:
            keep_versions: List of version numbers to keep
        """
        if not keep_versions:
            return
        
        # Sort and validate versions
        keep_versions = sorted(set(keep_versions))
        for v in keep_versions:
            if not (0 <= v < len(self.versions)):
                raise IndexError(f"Invalid version {v}")
        
        # Keep only specified versions
        new_versions = [self.versions[v] for v in keep_versions]
        self.versions = new_versions
    
    def __len__(self) -> int:
        return self.n
    
    def __str__(self) -> str:
        return f"PersistentSegmentTree(size={self.n}, versions={len(self.versions)})"


class PersistentRangeSum:
    """Persistent segment tree specialized for range sum queries."""
    
    def __init__(self, data: List[int]):
        self.tree = PersistentSegmentTree(data, lambda a, b: a + b, 0)
    
    def update(self, version: int, index: int, value: int) -> int:
        """Update value at index."""
        return self.tree.update(version, index, value)
    
    def range_sum(self, version: int, left: int, right: int) -> int:
        """Get sum of range [left, right]."""
        return self.tree.query(version, left, right)
    
    def point_query(self, version: int, index: int) -> int:
        """Get value at index."""
        return self.tree.point_query(version, index)
    
    def get_array(self, version: int) -> List[int]:
        """Get array representation."""
        return self.tree.get_array(version)


class PersistentRangeMin:
    """Persistent segment tree specialized for range minimum queries."""
    
    def __init__(self, data: List[int]):
        self.tree = PersistentSegmentTree(data, min, float('inf'))
    
    def update(self, version: int, index: int, value: int) -> int:
        """Update value at index."""
        return self.tree.update(version, index, value)
    
    def range_min(self, version: int, left: int, right: int) -> int:
        """Get minimum in range [left, right]."""
        return self.tree.query(version, left, right)
    
    def point_query(self, version: int, index: int) -> int:
        """Get value at index."""
        return self.tree.point_query(version, index)


class PersistentRangeMax:
    """Persistent segment tree specialized for range maximum queries."""
    
    def __init__(self, data: List[int]):
        self.tree = PersistentSegmentTree(data, max, float('-inf'))
    
    def update(self, version: int, index: int, value: int) -> int:
        """Update value at index."""
        return self.tree.update(version, index, value)
    
    def range_max(self, version: int, left: int, right: int) -> int:
        """Get maximum in range [left, right]."""
        return self.tree.query(version, left, right)
    
    def point_query(self, version: int, index: int) -> int:
        """Get value at index."""
        return self.tree.point_query(version, index)
