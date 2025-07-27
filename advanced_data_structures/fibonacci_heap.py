"""
Fibonacci Heap Implementation

Advanced priority queue with amortized O(1) operations for insert,
decrease-key, and merge. Optimal for algorithms like Dijkstra's shortest path.
"""

from typing import Optional, Any, List, Dict
import math


class FibonacciNode:
    """Node in a Fibonacci heap."""
    
    def __init__(self, key: float, data: Any = None):
        self.key = key
        self.data = data
        self.degree = 0
        self.marked = False
        self.parent: Optional['FibonacciNode'] = None
        self.child: Optional['FibonacciNode'] = None
        self.left: 'FibonacciNode' = self
        self.right: 'FibonacciNode' = self
    
    def add_child(self, child: 'FibonacciNode') -> None:
        """Add a child node."""
        if self.child is None:
            self.child = child
            child.left = child.right = child
        else:
            child.left = self.child
            child.right = self.child.right
            self.child.right.left = child
            self.child.right = child
        
        child.parent = self
        self.degree += 1
        child.marked = False
    
    def remove_child(self, child: 'FibonacciNode') -> None:
        """Remove a child node."""
        if self.degree == 1:
            self.child = None
        else:
            if self.child == child:
                self.child = child.right
            child.left.right = child.right
            child.right.left = child.left
        
        child.parent = None
        child.left = child.right = child
        self.degree -= 1


class FibonacciHeap:
    """
    Fibonacci heap implementation.
    
    Provides amortized O(1) for insert, find-min, decrease-key, and merge.
    Extract-min is O(log n) amortized.
    """
    
    def __init__(self):
        self.min_node: Optional[FibonacciNode] = None
        self.num_nodes = 0
        self.num_trees = 0
        self.num_marked = 0
    
    def is_empty(self) -> bool:
        """Check if heap is empty."""
        return self.min_node is None
    
    def size(self) -> int:
        """Return number of nodes in heap."""
        return self.num_nodes
    
    def insert(self, key: float, data: Any = None) -> FibonacciNode:
        """
        Insert new node with given key.
        
        Time Complexity: O(1) amortized
        
        Args:
            key: Priority key
            data: Optional data associated with key
            
        Returns:
            Reference to inserted node
        """
        node = FibonacciNode(key, data)
        
        if self.min_node is None:
            self.min_node = node
        else:
            self._add_to_root_list(node)
            if node.key < self.min_node.key:
                self.min_node = node
        
        self.num_nodes += 1
        self.num_trees += 1
        
        return node
    
    def find_min(self) -> Optional[FibonacciNode]:
        """
        Return minimum node without removing it.
        
        Time Complexity: O(1)
        """
        return self.min_node
    
    def extract_min(self) -> Optional[FibonacciNode]:
        """
        Remove and return minimum node.
        
        Time Complexity: O(log n) amortized
        """
        if self.min_node is None:
            return None
        
        min_node = self.min_node
        
        # Add all children of min_node to root list
        if min_node.child is not None:
            children = []
            child = min_node.child
            while True:
                children.append(child)
                child = child.right
                if child == min_node.child:
                    break
            
            for child in children:
                min_node.remove_child(child)
                self._add_to_root_list(child)
                self.num_trees += 1
        
        # Remove min_node from root list
        self._remove_from_root_list(min_node)
        self.num_trees -= 1
        self.num_nodes -= 1
        
        if self.num_nodes == 0:
            self.min_node = None
        else:
            self.min_node = min_node.right
            self._consolidate()
        
        return min_node
    
    def decrease_key(self, node: FibonacciNode, new_key: float) -> None:
        """
        Decrease key of given node.
        
        Time Complexity: O(1) amortized
        
        Args:
            node: Node to update
            new_key: New key value (must be <= current key)
            
        Raises:
            ValueError: If new_key > current key
        """
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        
        node.key = new_key
        parent = node.parent
        
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        
        if node.key < self.min_node.key:
            self.min_node = node
    
    def delete(self, node: FibonacciNode) -> None:
        """
        Delete given node from heap.
        
        Time Complexity: O(log n) amortized
        """
        self.decrease_key(node, float('-inf'))
        self.extract_min()
    
    def merge(self, other: 'FibonacciHeap') -> 'FibonacciHeap':
        """
        Merge with another Fibonacci heap.
        
        Time Complexity: O(1)
        
        Args:
            other: Another Fibonacci heap
            
        Returns:
            New merged heap
        """
        new_heap = FibonacciHeap()
        
        if self.min_node is None:
            new_heap.min_node = other.min_node
        elif other.min_node is None:
            new_heap.min_node = self.min_node
        else:
            # Merge root lists
            new_heap.min_node = self.min_node
            self.min_node.left.right = other.min_node.right
            other.min_node.right.left = self.min_node.left
            self.min_node.left = other.min_node
            other.min_node.right = self.min_node
            
            if other.min_node.key < self.min_node.key:
                new_heap.min_node = other.min_node
        
        new_heap.num_nodes = self.num_nodes + other.num_nodes
        new_heap.num_trees = self.num_trees + other.num_trees
        new_heap.num_marked = self.num_marked + other.num_marked
        
        return new_heap
    
    def _add_to_root_list(self, node: FibonacciNode) -> None:
        """Add node to root list."""
        if self.min_node is None:
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node
    
    def _remove_from_root_list(self, node: FibonacciNode) -> None:
        """Remove node from root list."""
        if node.right == node:
            # Only node in root list
            return
        
        node.left.right = node.right
        node.right.left = node.left
    
    def _consolidate(self) -> None:
        """Consolidate trees of same degree."""
        max_degree = int(math.log(self.num_nodes) * 2) + 1
        degree_table: List[Optional[FibonacciNode]] = [None] * max_degree
        
        # Collect all root nodes
        roots = []
        current = self.min_node
        while True:
            roots.append(current)
            current = current.right
            if current == self.min_node:
                break
        
        # Process each root
        for root in roots:
            degree = root.degree
            
            while degree_table[degree] is not None:
                other = degree_table[degree]
                
                if root.key > other.key:
                    root, other = other, root
                
                self._link(other, root)
                degree_table[degree] = None
                degree += 1
            
            degree_table[degree] = root
        
        # Find new minimum
        self.min_node = None
        self.num_trees = 0
        
        for node in degree_table:
            if node is not None:
                if self.min_node is None:
                    self.min_node = node
                    node.left = node.right = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node
                self.num_trees += 1
    
    def _link(self, child: FibonacciNode, parent: FibonacciNode) -> None:
        """Make child a child of parent."""
        # Remove child from root list
        child.left.right = child.right
        child.right.left = child.left
        
        # Add child to parent's child list
        parent.add_child(child)
        
        self.num_trees -= 1
    
    def _cut(self, child: FibonacciNode, parent: FibonacciNode) -> None:
        """Cut child from parent and add to root list."""
        parent.remove_child(child)
        self._add_to_root_list(child)
        child.marked = False
        self.num_trees += 1
        
        if child.marked:
            self.num_marked -= 1
    
    def _cascading_cut(self, node: FibonacciNode) -> None:
        """Perform cascading cut operation."""
        parent = node.parent
        
        if parent is not None:
            if not node.marked:
                node.marked = True
                self.num_marked += 1
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)
    
    def keys(self) -> List[float]:
        """Return all keys in heap (for debugging)."""
        if self.min_node is None:
            return []
        
        keys = []
        self._collect_keys(self.min_node, keys, set())
        return keys
    
    def _collect_keys(self, node: FibonacciNode, keys: List[float], visited: set) -> None:
        """Recursively collect all keys."""
        if node in visited:
            return
        
        visited.add(node)
        keys.append(node.key)
        
        # Visit children
        if node.child is not None:
            child = node.child
            while True:
                self._collect_keys(child, keys, visited)
                child = child.right
                if child == node.child:
                    break
        
        # Visit siblings in root list
        if node.parent is None:
            sibling = node.right
            while sibling != node and sibling not in visited:
                self._collect_keys(sibling, keys, visited)
                sibling = sibling.right
    
    def __len__(self) -> int:
        return self.num_nodes
    
    def __bool__(self) -> bool:
        return self.min_node is not None
    
    def __str__(self) -> str:
        if self.is_empty():
            return "FibonacciHeap(empty)"
        return f"FibonacciHeap(min={self.min_node.key}, size={self.num_nodes})"


class FibonacciPriorityQueue:
    """
    Priority queue implementation using Fibonacci heap.
    
    Provides better amortized complexity for decrease-key operations
    compared to binary heap, making it ideal for graph algorithms.
    """
    
    def __init__(self):
        self.heap = FibonacciHeap()
        self.entry_map: Dict[Any, FibonacciNode] = {}
        self._entry_count = 0
    
    def put(self, item: Any, priority: float) -> None:
        """
        Add item with given priority.
        
        Args:
            item: Item to add
            priority: Priority value (lower = higher priority)
        """
        if item in self.entry_map:
            # Update existing item
            self.decrease_priority(item, priority)
        else:
            # Insert new item
            node = self.heap.insert(priority, (self._entry_count, item))
            self.entry_map[item] = node
            self._entry_count += 1
    
    def get(self) -> Any:
        """Remove and return highest priority item."""
        if self.heap.is_empty():
            raise IndexError("get from empty priority queue")
        
        min_node = self.heap.extract_min()
        count, item = min_node.data
        del self.entry_map[item]
        return item
    
    def peek(self) -> Any:
        """Return highest priority item without removing."""
        if self.heap.is_empty():
            raise IndexError("peek from empty priority queue")
        
        min_node = self.heap.find_min()
        count, item = min_node.data
        return item
    
    def decrease_priority(self, item: Any, new_priority: float) -> None:
        """
        Decrease priority of existing item.
        
        Time Complexity: O(1) amortized
        
        Args:
            item: Item to update
            new_priority: New priority (must be <= current priority)
        """
        if item not in self.entry_map:
            raise KeyError(f"Item {item} not in priority queue")
        
        node = self.entry_map[item]
        self.heap.decrease_key(node, new_priority)
    
    def remove(self, item: Any) -> None:
        """Remove specific item from queue."""
        if item not in self.entry_map:
            raise KeyError(f"Item {item} not in priority queue")
        
        node = self.entry_map[item]
        self.heap.delete(node)
        del self.entry_map[item]
    
    def contains(self, item: Any) -> bool:
        """Check if item is in queue."""
        return item in self.entry_map
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.heap.is_empty()
    
    def size(self) -> int:
        """Return number of items in queue."""
        return self.heap.size()
    
    def clear(self) -> None:
        """Remove all items from queue."""
        self.heap = FibonacciHeap()
        self.entry_map.clear()
        self._entry_count = 0
    
    def __len__(self) -> int:
        return self.heap.size()
    
    def __bool__(self) -> bool:
        return not self.heap.is_empty()
    
    def __contains__(self, item: Any) -> bool:
        return self.contains(item)
