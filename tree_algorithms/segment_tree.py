"""
Segment Tree Implementation

A Segment Tree is a data structure that allows efficient range queries and updates on an array.
It's particularly useful for range sum queries, range minimum/maximum queries, and more.
"""

class SegmentTree:
    """
    Segment Tree implementation for range queries and updates.
    
    This implementation supports:
    - Range sum queries
    - Range minimum queries
    - Range maximum queries
    - Point updates
    """
    
    def __init__(self, arr, operation="sum"):
        """
        Initialize the segment tree with the given array.
        
        :param arr: The input array
        :param operation: The operation to perform ("sum", "min", "max")
        """
        self.n = len(arr)
        self.operation = operation
        
        # Choose the operation function
        if operation == "sum":
            self.op_func = lambda x, y: x + y
            self.default = 0
        elif operation == "min":
            self.op_func = min
            self.default = float('inf')
        elif operation == "max":
            self.op_func = max
            self.default = float('-inf')
        else:
            raise ValueError("Unsupported operation. Use 'sum', 'min', or 'max'")
        
        # Allocate memory for segment tree
        # Size of segment tree is 2*2^(ceil(log2(n))) - 1
        # We use a simplification: 4*n is always enough
        self.tree = [self.default] * (4 * self.n)
        
        # Build the segment tree
        if self.n > 0:
            self._build(arr, 0, self.n - 1, 0)
    
    def _build(self, arr, start, end, node):
        """
        Recursively build the segment tree.
        
        :param arr: Input array
        :param start: Start index of the current segment
        :param end: End index of the current segment
        :param node: Current node index in the segment tree
        :return: Value of the current node
        """
        if start == end:
            # Leaf node will have a single element
            self.tree[node] = arr[start]
            return self.tree[node]
        
        # Recursively build the left and right children
        mid = (start + end) // 2
        left_val = self._build(arr, start, mid, 2 * node + 1)
        right_val = self._build(arr, mid + 1, end, 2 * node + 2)
        
        # Internal node will hold the result of the operation
        self.tree[node] = self.op_func(left_val, right_val)
        
        return self.tree[node]
    
    def update(self, index, value):
        """
        Update the value at the given index in the array.
        
        :param index: Index in the original array to update
        :param value: New value to set
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds")
        
        self._update(0, self.n - 1, index, value, 0)
    
    def _update(self, start, end, index, value, node):
        """
        Recursively update the segment tree.
        
        :param start: Start index of the current segment
        :param end: End index of the current segment
        :param index: Index in the original array to update
        :param value: New value to set
        :param node: Current node index in the segment tree
        :return: Updated value of the current node
        """
        # If the input index is outside the range of this segment
        if index < start or index > end:
            return self.tree[node]
        
        # If this is a leaf node (base case)
        if start == end:
            self.tree[node] = value
            return value
        
        # Update the relevant child
        mid = (start + end) // 2
        if index <= mid:
            left_val = self._update(start, mid, index, value, 2 * node + 1)
            right_val = self.tree[2 * node + 2]
        else:
            left_val = self.tree[2 * node + 1]
            right_val = self._update(mid + 1, end, index, value, 2 * node + 2)
        
        # Update current node with the new values
        self.tree[node] = self.op_func(left_val, right_val)
        
        return self.tree[node]
    
    def query(self, left, right):
        """
        Query the segment tree for the result in the range [left, right].
        
        :param left: Left boundary of the range (inclusive)
        :param right: Right boundary of the range (inclusive)
        :return: Result of the operation in the given range
        """
        if left < 0 or right >= self.n or left > right:
            raise IndexError("Invalid query range")
        
        return self._query(0, self.n - 1, left, right, 0)
    
    def _query(self, start, end, left, right, node):
        """
        Recursively query the segment tree.
        
        :param start: Start index of the current segment
        :param end: End index of the current segment
        :param left: Left boundary of the query range
        :param right: Right boundary of the query range
        :param node: Current node index in the segment tree
        :return: Result of the operation in the given range
        """
        # If the current segment is completely outside the query range
        if start > right or end < left:
            return self.default
        
        # If the current segment is completely inside the query range
        if start >= left and end <= right:
            return self.tree[node]
        
        # If the current segment overlaps with the query range,
        # query both children and combine the results
        mid = (start + end) // 2
        left_val = self._query(start, mid, left, right, 2 * node + 1)
        right_val = self._query(mid + 1, end, left, right, 2 * node + 2)
        
        return self.op_func(left_val, right_val)
