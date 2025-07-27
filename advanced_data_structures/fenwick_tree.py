"""
Fenwick Tree (Binary Indexed Tree) Implementation

Efficient data structure for cumulative frequency tables and range sum queries.
Supports point updates and prefix sum queries in O(log n) time.
"""

from typing import List, Optional


class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for efficient range sum queries.
    
    Maintains cumulative frequencies and supports:
    - Point updates in O(log n)
    - Prefix sum queries in O(log n)
    - Range sum queries in O(log n)
    """
    
    def __init__(self, size: int):
        """
        Initialize Fenwick Tree with given size.
        
        Args:
            size: Maximum number of elements (1-indexed)
        """
        self.size = size
        self.tree = [0] * (size + 1)  # 1-indexed for easier bit manipulation
    
    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTree':
        """
        Create Fenwick Tree from existing array.
        
        Args:
            arr: Input array (0-indexed)
            
        Returns:
            FenwickTree initialized with array values
        """
        n = len(arr)
        ft = cls(n)
        
        for i, val in enumerate(arr):
            ft.update(i + 1, val)  # Convert to 1-indexed
        
        return ft
    
    def update(self, idx: int, delta: int) -> None:
        """
        Add delta to element at index idx.
        
        Time Complexity: O(log n)
        
        Args:
            idx: 1-indexed position to update
            delta: Value to add to current value
        """
        if idx <= 0 or idx > self.size:
            raise IndexError(f"Index {idx} out of range [1, {self.size}]")
        
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)  # Add least significant bit
    
    def prefix_sum(self, idx: int) -> int:
        """
        Get sum of elements from index 1 to idx (inclusive).
        
        Time Complexity: O(log n)
        
        Args:
            idx: 1-indexed position (inclusive)
            
        Returns:
            Sum of elements [1..idx]
        """
        if idx <= 0:
            return 0
        if idx > self.size:
            idx = self.size
        
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)  # Remove least significant bit
        
        return result
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Get sum of elements from left to right (inclusive, 1-indexed).
        
        Time Complexity: O(log n)
        
        Args:
            left: Start index (1-indexed, inclusive)
            right: End index (1-indexed, inclusive)
            
        Returns:
            Sum of elements [left..right]
        """
        if left > right:
            return 0
        
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def set_value(self, idx: int, value: int) -> None:
        """
        Set element at index idx to specific value.
        
        Time Complexity: O(log n)
        
        Args:
            idx: 1-indexed position
            value: New value to set
        """
        current_value = self.get_value(idx)
        self.update(idx, value - current_value)
    
    def get_value(self, idx: int) -> int:
        """
        Get value at specific index.
        
        Time Complexity: O(log n)
        
        Args:
            idx: 1-indexed position
            
        Returns:
            Value at position idx
        """
        return self.range_sum(idx, idx)
    
    def lower_bound(self, target_sum: int) -> int:
        """
        Find smallest index where prefix sum >= target_sum.
        
        Time Complexity: O(log n)
        
        Args:
            target_sum: Target cumulative sum
            
        Returns:
            Smallest 1-indexed position where prefix_sum >= target_sum,
            or size + 1 if no such position exists
        """
        if target_sum <= 0:
            return 1
        
        pos = 0
        bit_mask = 1
        
        # Find highest power of 2 <= size
        while bit_mask <= self.size:
            bit_mask <<= 1
        bit_mask >>= 1
        
        while bit_mask > 0:
            next_pos = pos + bit_mask
            if next_pos <= self.size and self.tree[next_pos] < target_sum:
                target_sum -= self.tree[next_pos]
                pos = next_pos
            bit_mask >>= 1
        
        return pos + 1
    
    def find_kth_element(self, k: int) -> int:
        """
        Find index of k-th element (1-indexed) in the frequency array.
        Assumes tree represents frequencies of elements.
        
        Time Complexity: O(log n)
        
        Args:
            k: Position of element to find (1-indexed)
            
        Returns:
            1-indexed position of k-th element
        """
        return self.lower_bound(k)
    
    def total_sum(self) -> int:
        """Get sum of all elements in the tree."""
        return self.prefix_sum(self.size)
    
    def clear(self) -> None:
        """Reset all values to 0."""
        self.tree = [0] * (self.size + 1)
    
    def to_array(self) -> List[int]:
        """
        Convert tree back to array form (0-indexed).
        
        Time Complexity: O(n log n)
        
        Returns:
            Array representation of current values
        """
        result = []
        for i in range(1, self.size + 1):
            result.append(self.get_value(i))
        return result
    
    def __len__(self) -> int:
        """Return size of the tree."""
        return self.size
    
    def __str__(self) -> str:
        """String representation of the tree."""
        values = self.to_array()
        return f"FenwickTree({values})"
    
    def __repr__(self) -> str:
        return f"FenwickTree(size={self.size}, sum={self.total_sum()})"


class FenwickTree2D:
    """
    2D Fenwick Tree for 2D range sum queries.
    
    Supports:
    - Point updates in O(log m × log n)
    - Rectangle sum queries in O(log m × log n)
    """
    
    def __init__(self, rows: int, cols: int):
        """
        Initialize 2D Fenwick Tree.
        
        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def update(self, row: int, col: int, delta: int) -> None:
        """
        Add delta to element at (row, col).
        
        Time Complexity: O(log m × log n)
        
        Args:
            row: 1-indexed row position
            col: 1-indexed column position
            delta: Value to add
        """
        if row <= 0 or row > self.rows or col <= 0 or col > self.cols:
            raise IndexError(f"Position ({row}, {col}) out of bounds")
        
        original_col = col
        while row <= self.rows:
            col = original_col
            while col <= self.cols:
                self.tree[row][col] += delta
                col += col & (-col)
            row += row & (-row)
    
    def prefix_sum(self, row: int, col: int) -> int:
        """
        Get sum of rectangle from (1,1) to (row, col).
        
        Time Complexity: O(log m × log n)
        """
        if row <= 0 or col <= 0:
            return 0
        
        row = min(row, self.rows)
        col = min(col, self.cols)
        
        result = 0
        original_col = col
        while row > 0:
            col = original_col
            while col > 0:
                result += self.tree[row][col]
                col -= col & (-col)
            row -= row & (-row)
        
        return result
    
    def range_sum(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Get sum of rectangle from (row1, col1) to (row2, col2) inclusive.
        
        Time Complexity: O(log m × log n)
        """
        if row1 > row2 or col1 > col2:
            return 0
        
        return (self.prefix_sum(row2, col2) - 
                self.prefix_sum(row1 - 1, col2) - 
                self.prefix_sum(row2, col1 - 1) + 
                self.prefix_sum(row1 - 1, col1 - 1))
    
    def set_value(self, row: int, col: int, value: int) -> None:
        """Set element at (row, col) to specific value."""
        current_value = self.get_value(row, col)
        self.update(row, col, value - current_value)
    
    def get_value(self, row: int, col: int) -> int:
        """Get value at specific position."""
        return self.range_sum(row, col, row, col)
    
    def __str__(self) -> str:
        """String representation of 2D tree."""
        return f"FenwickTree2D({self.rows}x{self.cols})"


class RangeFenwickTree:
    """
    Fenwick Tree supporting range updates and range queries.
    Uses difference array technique with two Fenwick trees.
    """
    
    def __init__(self, size: int):
        """Initialize range Fenwick tree."""
        self.size = size
        self.tree1 = FenwickTree(size)  # For difference array
        self.tree2 = FenwickTree(size)  # For prefix sums of difference array
    
    def range_update(self, left: int, right: int, delta: int) -> None:
        """
        Add delta to all elements in range [left, right].
        
        Time Complexity: O(log n)
        """
        self.tree1.update(left, delta)
        self.tree1.update(right + 1, -delta)
        self.tree2.update(left, delta * (left - 1))
        self.tree2.update(right + 1, -delta * right)
    
    def prefix_sum(self, idx: int) -> int:
        """Get prefix sum up to index idx."""
        return self.tree1.prefix_sum(idx) * idx - self.tree2.prefix_sum(idx)
    
    def range_sum(self, left: int, right: int) -> int:
        """Get sum of range [left, right]."""
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def point_update(self, idx: int, delta: int) -> None:
        """Add delta to single element."""
        self.range_update(idx, idx, delta)
