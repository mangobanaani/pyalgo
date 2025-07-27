"""
Binomial Heap Implementation

A collection of binomial trees with heap property.
Supports efficient merge operations and is used in priority queues.
"""

from typing import Optional, Any, List
import math


class BinomialNode:
    """Node in a binomial heap."""
    
    def __init__(self, key: Any, data: Any = None):
        self.key = key
        self.data = data
        self.degree = 0
        self.parent: Optional['BinomialNode'] = None
        self.child: Optional['BinomialNode'] = None
        self.sibling: Optional['BinomialNode'] = None
    
    def __str__(self) -> str:
        return f"BinomialNode(key={self.key}, degree={self.degree})"


class BinomialHeap:
    """
    Binomial Heap implementation.
    
    A collection of binomial trees where each tree satisfies min-heap property.
    Provides efficient merge operation and good amortized performance.
    """
    
    def __init__(self):
        self.head: Optional[BinomialNode] = None
        self._size = 0
    
    def is_empty(self) -> bool:
        """Check if heap is empty."""
        return self.head is None
    
    def size(self) -> int:
        """Return number of nodes in heap."""
        return self._size
    
    def insert(self, key: Any, data: Any = None) -> BinomialNode:
        """
        Insert new node with given key.
        
        Time Complexity: O(log n) amortized
        
        Args:
            key: Priority key
            data: Optional data associated with key
            
        Returns:
            Reference to inserted node
        """
        node = BinomialNode(key, data)
        temp_heap = BinomialHeap()
        temp_heap.head = node
        temp_heap._size = 1
        
        self._union(temp_heap)
        self._size += 1
        
        return node
    
    def find_min(self) -> Optional[BinomialNode]:
        """
        Find minimum node in heap.
        
        Time Complexity: O(log n)
        """
        if self.head is None:
            return None
        
        min_node = self.head
        current = self.head.sibling
        
        while current is not None:
            if current.key < min_node.key:
                min_node = current
            current = current.sibling
        
        return min_node
    
    def extract_min(self) -> Optional[BinomialNode]:
        """
        Remove and return minimum node.
        
        Time Complexity: O(log n)
        """
        if self.head is None:
            return None
        
        # Find minimum node and its predecessor
        min_node = self.head
        min_prev = None
        current = self.head
        current_prev = None
        
        while current.sibling is not None:
            if current.sibling.key < min_node.key:
                min_node = current.sibling
                min_prev = current
            current_prev = current
            current = current.sibling
        
        # Remove min_node from root list
        if min_prev is None:
            self.head = min_node.sibling
        else:
            min_prev.sibling = min_node.sibling
        
        # Create new heap from min_node's children
        child_heap = BinomialHeap()
        child = min_node.child
        
        if child is not None:
            # Reverse the order of children and make them roots
            children = []
            while child is not None:
                children.append(child)
                child = child.sibling
            
            # Reverse and set as roots
            for i in range(len(children) - 1, -1, -1):
                child = children[i]
                child.parent = None
                child.sibling = child_heap.head
                child_heap.head = child
        
        # Union with child heap
        self._union(child_heap)
        self._size -= 1
        
        return min_node
    
    def decrease_key(self, node: BinomialNode, new_key: Any) -> None:
        """
        Decrease key of given node.
        
        Time Complexity: O(log n)
        
        Args:
            node: Node to update
            new_key: New key value (must be <= current key)
            
        Raises:
            ValueError: If new_key > current key
        """
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        
        node.key = new_key
        current = node
        parent = current.parent
        
        # Bubble up to maintain heap property
        while parent is not None and current.key < parent.key:
            # Swap keys and data
            current.key, parent.key = parent.key, current.key
            current.data, parent.data = parent.data, current.data
            
            current = parent
            parent = current.parent
    
    def delete(self, node: BinomialNode) -> None:
        """
        Delete given node from heap.
        
        Time Complexity: O(log n)
        """
        self.decrease_key(node, float('-inf'))
        self.extract_min()
    
    def merge(self, other: 'BinomialHeap') -> 'BinomialHeap':
        """
        Merge with another binomial heap.
        
        Time Complexity: O(log n)
        
        Args:
            other: Another binomial heap
            
        Returns:
            New merged heap
        """
        new_heap = BinomialHeap()
        new_heap.head = self._merge_root_lists(self.head, other.head)
        new_heap._size = self._size + other._size
        
        if new_heap.head is not None:
            new_heap._consolidate()
        
        return new_heap
    
    def _union(self, other: 'BinomialHeap') -> None:
        """Union this heap with another heap (modifies this heap)."""
        self.head = self._merge_root_lists(self.head, other.head)
        
        if self.head is not None:
            self._consolidate()
    
    def _merge_root_lists(self, h1: Optional[BinomialNode], 
                         h2: Optional[BinomialNode]) -> Optional[BinomialNode]:
        """Merge two root lists in order of increasing degree."""
        if h1 is None:
            return h2
        if h2 is None:
            return h1
        
        # Determine head of merged list
        if h1.degree <= h2.degree:
            head = h1
            h1 = h1.sibling
        else:
            head = h2
            h2 = h2.sibling
        
        current = head
        
        # Merge remaining nodes
        while h1 is not None and h2 is not None:
            if h1.degree <= h2.degree:
                current.sibling = h1
                h1 = h1.sibling
            else:
                current.sibling = h2
                h2 = h2.sibling
            current = current.sibling
        
        # Attach remaining nodes
        if h1 is not None:
            current.sibling = h1
        else:
            current.sibling = h2
        
        return head
    
    def _consolidate(self) -> None:
        """Consolidate trees of same degree."""
        if self.head is None:
            return
        
        prev = None
        current = self.head
        next_node = current.sibling
        
        while next_node is not None:
            if (current.degree != next_node.degree or
                (next_node.sibling is not None and 
                 next_node.sibling.degree == current.degree)):
                # Case 1: Degrees different or three consecutive trees of same degree
                prev = current
                current = next_node
            elif current.key <= next_node.key:
                # Case 2: current becomes parent of next_node
                current.sibling = next_node.sibling
                self._link(next_node, current)
            else:
                # Case 3: next_node becomes parent of current
                if prev is None:
                    self.head = next_node
                else:
                    prev.sibling = next_node
                
                self._link(current, next_node)
                current = next_node
            
            next_node = current.sibling
    
    def _link(self, child: BinomialNode, parent: BinomialNode) -> None:
        """Make child a child of parent."""
        child.parent = parent
        child.sibling = parent.child
        parent.child = child
        parent.degree += 1
    
    def keys(self) -> List[Any]:
        """Return all keys in heap (for debugging)."""
        keys = []
        self._collect_keys(self.head, keys)
        return keys
    
    def _collect_keys(self, node: Optional[BinomialNode], keys: List[Any]) -> None:
        """Recursively collect all keys."""
        while node is not None:
            keys.append(node.key)
            self._collect_keys(node.child, keys)
            node = node.sibling
    
    def print_heap(self) -> None:
        """Print heap structure (for debugging)."""
        if self.head is None:
            print("Empty heap")
            return
        
        print("Binomial Heap:")
        current = self.head
        tree_index = 0
        
        while current is not None:
            print(f"Tree {tree_index} (degree {current.degree}):")
            self._print_tree(current, 0)
            current = current.sibling
            tree_index += 1
    
    def _print_tree(self, node: Optional[BinomialNode], depth: int) -> None:
        """Print binomial tree structure."""
        if node is not None:
            print("  " * depth + str(node.key))
            child = node.child
            while child is not None:
                self._print_tree(child, depth + 1)
                child = child.sibling
    
    def validate(self) -> bool:
        """
        Validate binomial heap properties.
        
        Returns:
            True if heap satisfies all binomial heap properties
        """
        if self.head is None:
            return True
        
        # Check if root list is sorted by degree
        current = self.head
        while current.sibling is not None:
            if current.degree > current.sibling.degree:
                return False
            if current.degree == current.sibling.degree:
                return False  # No two trees should have same degree
            current = current.sibling
        
        # Check each binomial tree
        current = self.head
        while current is not None:
            if not self._validate_tree(current):
                return False
            current = current.sibling
        
        return True
    
    def _validate_tree(self, node: BinomialNode) -> bool:
        """Validate single binomial tree."""
        # Check heap property
        child = node.child
        while child is not None:
            if child.key < node.key:
                return False
            if not self._validate_tree(child):
                return False
            child = child.sibling
        
        # Check degree property
        expected_degree = 0
        child = node.child
        while child is not None:
            expected_degree += 1
            child = child.sibling
        
        return expected_degree == node.degree
    
    def clear(self) -> None:
        """Remove all nodes from heap."""
        self.head = None
        self._size = 0
    
    def __len__(self) -> int:
        return self._size
    
    def __bool__(self) -> bool:
        return self.head is not None
    
    def __str__(self) -> str:
        if self.is_empty():
            return "BinomialHeap(empty)"
        min_node = self.find_min()
        return f"BinomialHeap(min={min_node.key if min_node else None}, size={self._size})"


class BinomialPriorityQueue:
    """
    Priority queue implementation using binomial heap.
    
    Provides efficient merge operations, making it suitable for
    algorithms that need to merge priority queues.
    """
    
    def __init__(self):
        self.heap = BinomialHeap()
        self._entry_count = 0
    
    def put(self, item: Any, priority: float) -> None:
        """
        Add item with given priority.
        
        Args:
            item: Item to add
            priority: Priority value (lower = higher priority)
        """
        self.heap.insert(priority, (self._entry_count, item))
        self._entry_count += 1
    
    def get(self) -> Any:
        """Remove and return highest priority item."""
        if self.heap.is_empty():
            raise IndexError("get from empty priority queue")
        
        min_node = self.heap.extract_min()
        count, item = min_node.data
        return item
    
    def peek(self) -> Any:
        """Return highest priority item without removing."""
        if self.heap.is_empty():
            raise IndexError("peek from empty priority queue")
        
        min_node = self.heap.find_min()
        count, item = min_node.data
        return item
    
    def merge(self, other: 'BinomialPriorityQueue') -> 'BinomialPriorityQueue':
        """
        Merge with another priority queue.
        
        Time Complexity: O(log n)
        
        Args:
            other: Another binomial priority queue
            
        Returns:
            New merged priority queue
        """
        new_pq = BinomialPriorityQueue()
        new_pq.heap = self.heap.merge(other.heap)
        new_pq._entry_count = max(self._entry_count, other._entry_count) + 1
        return new_pq
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.heap.is_empty()
    
    def size(self) -> int:
        """Return number of items in queue."""
        return self.heap.size()
    
    def clear(self) -> None:
        """Remove all items from queue."""
        self.heap.clear()
        self._entry_count = 0
    
    def __len__(self) -> int:
        return self.heap.size()
    
    def __bool__(self) -> bool:
        return not self.heap.is_empty()
