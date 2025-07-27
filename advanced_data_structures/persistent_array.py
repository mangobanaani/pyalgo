"""
Persistent Array Implementation

Immutable array with efficient copy-on-write operations.
Supports functional programming paradigms with persistent data structures.
"""

from typing import Any, Optional, List, Iterator
import copy


class PersistentArrayNode:
    """Node in a persistent array tree structure."""
    
    def __init__(self, value: Any = None, is_leaf: bool = True):
        self.value = value
        self.is_leaf = is_leaf
        self.children: List[Optional['PersistentArrayNode']] = []
        self.version = 0
    
    def copy(self) -> 'PersistentArrayNode':
        """Create a shallow copy of the node."""
        new_node = PersistentArrayNode(self.value, self.is_leaf)
        new_node.children = self.children.copy()
        new_node.version = self.version + 1
        return new_node


class PersistentArray:
    """
    Persistent Array implementation using tree structure.
    
    Provides O(log n) access, update, and append operations while
    maintaining immutability. Previous versions remain accessible.
    """
    
    BRANCHING_FACTOR = 32  # Common choice for persistent data structures
    
    def __init__(self, data: Optional[List[Any]] = None):
        self._size = 0
        self._depth = 0
        self._root: Optional[PersistentArrayNode] = None
        self._version = 0
        
        if data:
            self._build_from_list(data)
    
    def _build_from_list(self, data: List[Any]) -> None:
        """Build persistent array from list."""
        if not data:
            return
        
        self._size = len(data)
        
        # Calculate required depth
        self._depth = 0
        temp_size = len(data)
        while temp_size > self.BRANCHING_FACTOR:
            temp_size = (temp_size + self.BRANCHING_FACTOR - 1) // self.BRANCHING_FACTOR
            self._depth += 1
        
        # Build tree bottom-up
        current_level = []
        
        # Create leaf nodes
        for value in data:
            leaf = PersistentArrayNode(value, True)
            current_level.append(leaf)
        
        # Build internal nodes
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), self.BRANCHING_FACTOR):
                internal = PersistentArrayNode(None, False)
                internal.children = current_level[i:i + self.BRANCHING_FACTOR]
                next_level.append(internal)
            
            current_level = next_level
        
        self._root = current_level[0] if current_level else None
    
    def get(self, index: int) -> Any:
        """
        Get value at given index.
        
        Time Complexity: O(log n)
        
        Args:
            index: Index to access
            
        Returns:
            Value at index
            
        Raises:
            IndexError: If index is out of bounds
        """
        if not (0 <= index < self._size):
            raise IndexError(f"Index {index} out of range [0, {self._size})")
        
        return self._get_recursive(self._root, index, self._depth)
    
    def _get_recursive(self, node: Optional[PersistentArrayNode], 
                      index: int, depth: int) -> Any:
        """Recursively get value at index."""
        if node is None:
            raise IndexError("Invalid index")
        
        if node.is_leaf:
            return node.value
        
        # Calculate which child contains the index
        child_size = self.BRANCHING_FACTOR ** depth
        child_index = index // child_size
        remaining_index = index % child_size
        
        if child_index >= len(node.children):
            raise IndexError("Invalid index")
        
        return self._get_recursive(node.children[child_index], 
                                 remaining_index, depth - 1)
    
    def set(self, index: int, value: Any) -> 'PersistentArray':
        """
        Create new array with value set at index.
        
        Time Complexity: O(log n)
        
        Args:
            index: Index to set
            value: New value
            
        Returns:
            New persistent array with updated value
            
        Raises:
            IndexError: If index is out of bounds
        """
        if not (0 <= index < self._size):
            raise IndexError(f"Index {index} out of range [0, {self._size})")
        
        new_array = PersistentArray()
        new_array._size = self._size
        new_array._depth = self._depth
        new_array._version = self._version + 1
        new_array._root = self._set_recursive(self._root, index, value, self._depth)
        
        return new_array
    
    def _set_recursive(self, node: Optional[PersistentArrayNode], 
                      index: int, value: Any, depth: int) -> PersistentArrayNode:
        """Recursively create new nodes with updated value."""
        if node is None:
            raise IndexError("Invalid index")
        
        new_node = node.copy()
        
        if new_node.is_leaf:
            new_node.value = value
            return new_node
        
        # Calculate which child contains the index
        child_size = self.BRANCHING_FACTOR ** depth
        child_index = index // child_size
        remaining_index = index % child_size
        
        if child_index >= len(new_node.children):
            raise IndexError("Invalid index")
        
        # Update the specific child
        new_node.children[child_index] = self._set_recursive(
            new_node.children[child_index], remaining_index, value, depth - 1)
        
        return new_node
    
    def append(self, value: Any) -> 'PersistentArray':
        """
        Create new array with value appended.
        
        Time Complexity: O(log n) amortized
        
        Args:
            value: Value to append
            
        Returns:
            New persistent array with appended value
        """
        new_array = PersistentArray()
        new_array._size = self._size + 1
        new_array._version = self._version + 1
        
        # Check if we need to increase depth
        if self._size == 0:
            new_array._depth = 0
            new_array._root = PersistentArrayNode(value, True)
        elif self._size < self.BRANCHING_FACTOR ** (self._depth + 1):
            # Can fit in current depth
            new_array._depth = self._depth
            new_array._root = self._append_recursive(self._root, value, self._depth)
        else:
            # Need to increase depth
            new_array._depth = self._depth + 1
            new_root = PersistentArrayNode(None, False)
            new_root.children = [self._root]
            new_array._root = self._append_recursive(new_root, value, new_array._depth)
        
        return new_array
    
    def _append_recursive(self, node: Optional[PersistentArrayNode], 
                         value: Any, depth: int) -> PersistentArrayNode:
        """Recursively append value to array."""
        if depth == 0:
            # Create new leaf node
            return PersistentArrayNode(value, True)
        
        new_node = node.copy() if node else PersistentArrayNode(None, False)
        
        # Calculate current capacity and used slots
        child_capacity = self.BRANCHING_FACTOR ** depth
        used_slots = (self._size + child_capacity - 1) // child_capacity
        
        if used_slots > len(new_node.children):
            # Need to add new child
            new_child = self._append_recursive(None, value, depth - 1)
            new_node.children.append(new_child)
        else:
            # Append to last child
            last_child = new_node.children[-1] if new_node.children else None
            new_node.children[-1] = self._append_recursive(last_child, value, depth - 1)
        
        return new_node
    
    def pop(self) -> tuple['PersistentArray', Any]:
        """
        Create new array with last element removed.
        
        Time Complexity: O(log n)
        
        Returns:
            Tuple of (new_array, popped_value)
            
        Raises:
            IndexError: If array is empty
        """
        if self._size == 0:
            raise IndexError("pop from empty array")
        
        popped_value = self.get(self._size - 1)
        
        new_array = PersistentArray()
        new_array._size = self._size - 1
        new_array._version = self._version + 1
        
        if new_array._size == 0:
            new_array._depth = 0
            new_array._root = None
        else:
            new_array._depth = self._depth
            new_array._root = self._pop_recursive(self._root, self._depth)
        
        return new_array, popped_value
    
    def _pop_recursive(self, node: Optional[PersistentArrayNode], 
                      depth: int) -> Optional[PersistentArrayNode]:
        """Recursively remove last element."""
        if node is None:
            return None
        
        if depth == 0:
            # Leaf node - remove it
            return None
        
        new_node = node.copy()
        
        # Remove from last child
        if new_node.children:
            new_node.children[-1] = self._pop_recursive(new_node.children[-1], depth - 1)
            
            # Remove empty children
            while new_node.children and new_node.children[-1] is None:
                new_node.children.pop()
        
        # Return None if node becomes empty
        return new_node if new_node.children else None
    
    def slice(self, start: int, stop: int) -> 'PersistentArray':
        """
        Create new array with elements from start to stop.
        
        Args:
            start: Start index (inclusive)
            stop: Stop index (exclusive)
            
        Returns:
            New persistent array with sliced elements
        """
        if start < 0:
            start = max(0, self._size + start)
        if stop < 0:
            stop = max(0, self._size + stop)
        
        start = max(0, min(start, self._size))
        stop = max(start, min(stop, self._size))
        
        if start == stop:
            return PersistentArray()
        
        # Extract elements
        elements = []
        for i in range(start, stop):
            elements.append(self.get(i))
        
        return PersistentArray(elements)
    
    def concat(self, other: 'PersistentArray') -> 'PersistentArray':
        """
        Create new array by concatenating with another array.
        
        Args:
            other: Array to concatenate
            
        Returns:
            New persistent array with concatenated elements
        """
        # Simple implementation - convert to lists and rebuild
        elements = self.to_list() + other.to_list()
        return PersistentArray(elements)
    
    def to_list(self) -> List[Any]:
        """Convert to regular Python list."""
        result = []
        for i in range(self._size):
            result.append(self.get(i))
        return result
    
    def size(self) -> int:
        """Return number of elements in array."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if array is empty."""
        return self._size == 0
    
    def version(self) -> int:
        """Return version number of this array."""
        return self._version
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, key) -> Any:
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            return self.slice(start, stop)
        else:
            return self.get(key)
    
    def __iter__(self) -> Iterator[Any]:
        for i in range(self._size):
            yield self.get(i)
    
    def __bool__(self) -> bool:
        return self._size > 0
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PersistentArray):
            return False
        
        if self._size != other._size:
            return False
        
        for i in range(self._size):
            if self.get(i) != other.get(i):
                return False
        
        return True
    
    def __str__(self) -> str:
        if self._size <= 10:
            elements = ", ".join(str(self.get(i)) for i in range(self._size))
            return f"PersistentArray([{elements}])"
        else:
            first_few = ", ".join(str(self.get(i)) for i in range(3))
            last_few = ", ".join(str(self.get(i)) for i in range(self._size - 3, self._size))
            return f"PersistentArray([{first_few}, ..., {last_few}])"
    
    def __repr__(self) -> str:
        return self.__str__()


class PersistentVector(PersistentArray):
    """
    Alias for PersistentArray with vector-like interface.
    
    Provides functional programming interface for immutable sequences.
    """
    
    def push(self, value: Any) -> 'PersistentVector':
        """Add element to end (alias for append)."""
        result = self.append(value)
        return PersistentVector._from_persistent_array(result)
    
    def peek(self) -> Any:
        """Get last element without removing."""
        if self._size == 0:
            raise IndexError("peek from empty vector")
        return self.get(self._size - 1)
    
    def pop_back(self) -> tuple['PersistentVector', Any]:
        """Remove last element (alias for pop)."""
        new_array, value = self.pop()
        return PersistentVector._from_persistent_array(new_array), value
    
    @classmethod
    def _from_persistent_array(cls, array: PersistentArray) -> 'PersistentVector':
        """Create PersistentVector from PersistentArray."""
        vector = cls()
        vector._size = array._size
        vector._depth = array._depth
        vector._root = array._root
        vector._version = array._version
        return vector
    
    @classmethod
    def empty(cls) -> 'PersistentVector':
        """Create empty persistent vector."""
        return cls()
    
    @classmethod
    def from_iterable(cls, iterable) -> 'PersistentVector':
        """Create persistent vector from iterable."""
        return cls(list(iterable))
