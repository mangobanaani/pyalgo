"""
Skip List implementation.

A skip list is a probabilistic data structure that allows O(log n) search,
insertion, and deletion operations in average case. It's an alternative to
balanced trees with simpler implementation.
"""

import random
from typing import Optional, List, TypeVar

T = TypeVar('T')


class SkipNode:
    """Node in a skip list."""
    
    def __init__(self, value: T, level: int):
        self.value = value
        self.forward: List[Optional['SkipNode']] = [None] * (level + 1)


class SkipList:
    """
    Skip List data structure.
    
    A probabilistic alternative to balanced trees that provides O(log n)
    expected time for search, insertion, and deletion operations.
    """
    
    def __init__(self, max_level: int = 16, p: float = 0.5):
        """
        Initialize skip list.
        
        Args:
            max_level: Maximum number of levels
            p: Probability for level promotion (typically 0.5)
        """
        self.max_level = max_level
        self.p = p
        self.level = 0
        
        # Create header node with maximum level
        self.header = SkipNode(None, max_level)
    
    def _random_level(self) -> int:
        """Generate random level for new node."""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def search(self, target: T) -> bool:
        """
        Search for a value in the skip list.
        
        Args:
            target: Value to search for
            
        Returns:
            True if value is found, False otherwise
            
        Time Complexity: O(log n) expected
        """
        current = self.header
        
        # Start from highest level and go down
        for level in range(self.level, -1, -1):
            # Move forward while next node value is less than target
            while (current.forward[level] is not None and 
                   current.forward[level].value < target):
                current = current.forward[level]
        
        # Move to next node at level 0
        current = current.forward[0]
        
        # Check if we found the target
        return current is not None and current.value == target
    
    def insert(self, value: T) -> None:
        """
        Insert a value into the skip list.
        
        Args:
            value: Value to insert
            
        Time Complexity: O(log n) expected
        """
        # Keep track of update array
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find insertion point
        for level in range(self.level, -1, -1):
            while (current.forward[level] is not None and 
                   current.forward[level].value < value):
                current = current.forward[level]
            update[level] = current
        
        # Move to next node at level 0
        current = current.forward[0]
        
        # If value doesn't exist, insert it
        if current is None or current.value != value:
            # Generate random level for new node
            new_level = self._random_level()
            
            # If new level is greater than current level, update header links
            if new_level > self.level:
                for level in range(self.level + 1, new_level + 1):
                    update[level] = self.header
                self.level = new_level
            
            # Create new node and insert
            new_node = SkipNode(value, new_level)
            for level in range(new_level + 1):
                new_node.forward[level] = update[level].forward[level]
                update[level].forward[level] = new_node
    
    def delete(self, value: T) -> bool:
        """
        Delete a value from the skip list.
        
        Args:
            value: Value to delete
            
        Returns:
            True if value was found and deleted, False otherwise
            
        Time Complexity: O(log n) expected
        """
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find the node to delete
        for level in range(self.level, -1, -1):
            while (current.forward[level] is not None and 
                   current.forward[level].value < value):
                current = current.forward[level]
            update[level] = current
        
        current = current.forward[0]
        
        # If node is found, delete it
        if current is not None and current.value == value:
            # Update forward links
            for level in range(self.level + 1):
                if update[level].forward[level] != current:
                    break
                update[level].forward[level] = current.forward[level]
            
            # Update skip list level
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1
            
            return True
        
        return False
    
    def get_all_values(self) -> List[T]:
        """
        Get all values in the skip list in sorted order.
        
        Returns:
            List of all values in ascending order
            
        Time Complexity: O(n)
        """
        values = []
        current = self.header.forward[0]
        
        while current is not None:
            values.append(current.value)
            current = current.forward[0]
        
        return values
    
    def display(self) -> None:
        """Display the skip list structure (for debugging)."""
        print("Skip List:")
        for level in range(self.level, -1, -1):
            print(f"Level {level}: ", end="")
            current = self.header.forward[level]
            while current is not None:
                print(f"{current.value} ", end="")
                current = current.forward[level]
            print()
    
    def __len__(self) -> int:
        """Return the number of elements in the skip list."""
        count = 0
        current = self.header.forward[0]
        while current is not None:
            count += 1
            current = current.forward[0]
        return count
    
    def __contains__(self, value: T) -> bool:
        """Check if value is in skip list."""
        return self.search(value)
