"""
Binary Heap Implementation

Priority queue data structure implemented as a complete binary tree.
Supports both min-heap and max-heap variants with efficient operations.
"""

from typing import List, Optional, Callable, Any
import heapq


class BinaryHeap:
    """
    Generic binary heap implementation.
    
    Can be configured as min-heap or max-heap using comparison function.
    Maintains heap property: parent is always smaller/larger than children.
    """
    
    def __init__(self, is_min_heap: bool = True, key_func: Optional[Callable] = None):
        """
        Initialize binary heap.
        
        Args:
            is_min_heap: True for min-heap, False for max-heap
            key_func: Optional function to extract comparison key from elements
        """
        self.heap: List[Any] = []
        self.is_min_heap = is_min_heap
        self.key_func = key_func or (lambda x: x)
        
    def _compare(self, a: Any, b: Any) -> bool:
        """Compare two elements based on heap type."""
        key_a, key_b = self.key_func(a), self.key_func(b)
        return key_a <= key_b if self.is_min_heap else key_a >= key_b
    
    def _parent_index(self, index: int) -> int:
        """Get parent index."""
        return (index - 1) // 2
    
    def _left_child_index(self, index: int) -> int:
        """Get left child index."""
        return 2 * index + 1
    
    def _right_child_index(self, index: int) -> int:
        """Get right child index."""
        return 2 * index + 2
    
    def _has_parent(self, index: int) -> bool:
        """Check if node has parent."""
        return self._parent_index(index) >= 0
    
    def _has_left_child(self, index: int) -> bool:
        """Check if node has left child."""
        return self._left_child_index(index) < len(self.heap)
    
    def _has_right_child(self, index: int) -> bool:
        """Check if node has right child."""
        return self._right_child_index(index) < len(self.heap)
    
    def _parent(self, index: int) -> Any:
        """Get parent element."""
        return self.heap[self._parent_index(index)]
    
    def _left_child(self, index: int) -> Any:
        """Get left child element."""
        return self.heap[self._left_child_index(index)]
    
    def _right_child(self, index: int) -> Any:
        """Get right child element."""
        return self.heap[self._right_child_index(index)]
    
    def _swap(self, index1: int, index2: int) -> None:
        """Swap elements at two indices."""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
    
    def insert(self, element: Any) -> None:
        """
        Insert element into heap.
        
        Time Complexity: O(log n)
        """
        self.heap.append(element)
        self._heapify_up()
    
    def _heapify_up(self) -> None:
        """Restore heap property by moving element up."""
        index = len(self.heap) - 1
        while (self._has_parent(index) and 
               not self._compare(self._parent(index), self.heap[index])):
            self._swap(self._parent_index(index), index)
            index = self._parent_index(index)
    
    def extract(self) -> Any:
        """
        Remove and return root element (min/max).
        
        Time Complexity: O(log n)
        
        Returns:
            Root element
            
        Raises:
            IndexError: If heap is empty
        """
        if not self.heap:
            raise IndexError("extract from empty heap")
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down()
        return root
    
    def _heapify_down(self) -> None:
        """Restore heap property by moving element down."""
        index = 0
        while self._has_left_child(index):
            target_child_index = self._get_target_child_index(index)
            
            if self._compare(self.heap[index], self.heap[target_child_index]):
                break
            
            self._swap(index, target_child_index)
            index = target_child_index
    
    def _get_target_child_index(self, index: int) -> int:
        """Get index of child to potentially swap with."""
        if not self._has_right_child(index):
            return self._left_child_index(index)
        
        left_index = self._left_child_index(index)
        right_index = self._right_child_index(index)
        
        return (left_index if self._compare(self.heap[left_index], self.heap[right_index])
                else right_index)
    
    def peek(self) -> Any:
        """
        Return root element without removing it.
        
        Time Complexity: O(1)
        """
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]
    
    def size(self) -> int:
        """Return number of elements in heap."""
        return len(self.heap)
    
    def is_empty(self) -> bool:
        """Check if heap is empty."""
        return len(self.heap) == 0
    
    def clear(self) -> None:
        """Remove all elements from heap."""
        self.heap.clear()
    
    @classmethod
    def heapify(cls, arr: List[Any], is_min_heap: bool = True, key_func: Optional[Callable] = None) -> 'BinaryHeap':
        """
        Create heap from existing array using Floyd's algorithm.
        
        Time Complexity: O(n)
        
        Args:
            arr: Input array
            is_min_heap: True for min-heap, False for max-heap
            key_func: Optional key extraction function
            
        Returns:
            New BinaryHeap containing all elements from arr
        """
        heap = cls(is_min_heap, key_func)
        heap.heap = arr.copy()
        
        # Start from last non-leaf node and heapify down
        for i in range(len(arr) // 2 - 1, -1, -1):
            heap._heapify_down_from(i)
        
        return heap
    
    def _heapify_down_from(self, start_index: int) -> None:
        """Heapify down starting from specific index."""
        index = start_index
        while self._has_left_child(index):
            target_child_index = self._get_target_child_index(index)
            
            if self._compare(self.heap[index], self.heap[target_child_index]):
                break
            
            self._swap(index, target_child_index)
            index = target_child_index
    
    def to_list(self) -> List[Any]:
        """Return copy of internal heap array."""
        return self.heap.copy()
    
    def __len__(self) -> int:
        return len(self.heap)
    
    def __bool__(self) -> bool:
        return len(self.heap) > 0
    
    def __str__(self) -> str:
        heap_type = "MinHeap" if self.is_min_heap else "MaxHeap"
        return f"{heap_type}({self.heap})"
    
    def __repr__(self) -> str:
        return self.__str__()


class MinHeap(BinaryHeap):
    """Min-heap specialization of BinaryHeap."""
    
    def __init__(self, key_func: Optional[Callable] = None):
        super().__init__(is_min_heap=True, key_func=key_func)
    
    def get_min(self) -> Any:
        """Get minimum element (alias for peek)."""
        return self.peek()
    
    def extract_min(self) -> Any:
        """Extract minimum element (alias for extract)."""
        return self.extract()


class MaxHeap(BinaryHeap):
    """Max-heap specialization of BinaryHeap."""
    
    def __init__(self, key_func: Optional[Callable] = None):
        super().__init__(is_min_heap=False, key_func=key_func)
    
    def get_max(self) -> Any:
        """Get maximum element (alias for peek)."""
        return self.peek()
    
    def extract_max(self) -> Any:
        """Extract maximum element (alias for extract)."""
        return self.extract()


class HeapSort:
    """Heap sort algorithm using binary heap."""
    
    @staticmethod
    def sort(arr: List[Any], reverse: bool = False, key_func: Optional[Callable] = None) -> List[Any]:
        """
        Sort array using heap sort algorithm.
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Args:
            arr: Array to sort
            reverse: True for descending order
            key_func: Optional key extraction function
            
        Returns:
            New sorted array
        """
        if not arr:
            return []
        
        # Use opposite heap type for desired sort order
        heap = BinaryHeap(is_min_heap=not reverse, key_func=key_func)
        
        # Insert all elements
        for element in arr:
            heap.insert(element)
        
        # Extract all elements in sorted order
        result = []
        while not heap.is_empty():
            result.append(heap.extract())
        
        return result
    
    @staticmethod
    def sort_in_place(arr: List[Any], reverse: bool = False, key_func: Optional[Callable] = None) -> None:
        """
        Sort array in place using heap sort.
        
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        if len(arr) <= 1:
            return
        
        # Build heap in place
        heap = BinaryHeap(is_min_heap=reverse, key_func=key_func)
        heap.heap = arr
        
        # Heapify the array
        for i in range(len(arr) // 2 - 1, -1, -1):
            heap._heapify_down_from(i)
        
        # Extract elements one by one
        for i in range(len(arr) - 1, 0, -1):
            # Move current root to end
            arr[0], arr[i] = arr[i], arr[0]
            
            # Reduce heap size and heapify
            heap.heap = arr[:i]
            heap._heapify_down_from(0)


class PriorityQueue:
    """
    Priority queue implementation using binary heap.
    
    Supports elements with priorities, where lower priority values
    indicate higher priority (min-heap behavior).
    """
    
    def __init__(self):
        self.heap = MinHeap(key_func=lambda x: x[0])  # Priority is first element
        self._entry_count = 0
    
    def put(self, item: Any, priority: float = 0) -> None:
        """
        Add item with given priority.
        
        Args:
            item: Item to add
            priority: Priority value (lower = higher priority)
        """
        # Use entry count as tiebreaker to maintain FIFO for equal priorities
        entry = (priority, self._entry_count, item)
        self.heap.insert(entry)
        self._entry_count += 1
    
    def get(self) -> Any:
        """Remove and return highest priority item."""
        if self.heap.is_empty():
            raise IndexError("get from empty priority queue")
        
        priority, count, item = self.heap.extract()
        return item
    
    def peek(self) -> Any:
        """Return highest priority item without removing."""
        if self.heap.is_empty():
            raise IndexError("peek from empty priority queue")
        
        priority, count, item = self.heap.peek()
        return item
    
    def empty(self) -> bool:
        """Check if priority queue is empty."""
        return self.heap.is_empty()
    
    def size(self) -> int:
        """Return number of items in queue."""
        return self.heap.size()
    
    def __len__(self) -> int:
        return self.heap.size()
    
    def __bool__(self) -> bool:
        return not self.heap.is_empty()
