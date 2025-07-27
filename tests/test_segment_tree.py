import pytest
from tree_algorithms.segment_tree import SegmentTree

def test_segment_tree_sum():
    """Test segment tree with sum operation."""
    # Basic test case
    arr = [1, 3, 5, 7, 9, 11]
    tree = SegmentTree(arr, "sum")
    
    # Test range queries
    assert tree.query(0, 5) == 36  # Sum of entire array
    assert tree.query(1, 3) == 15  # Sum of [3, 5, 7]
    assert tree.query(0, 0) == 1  # Single element
    assert tree.query(5, 5) == 11  # Single element at the end
    
    # Test updates
    tree.update(1, 10)  # Change 3 to 10
    assert tree.query(0, 5) == 43  # New sum after update
    assert tree.query(1, 3) == 22  # New sum of subrange after update

def test_segment_tree_min():
    """Test segment tree with minimum operation."""
    # Basic test case
    arr = [5, 2, 8, 1, 9, 3]
    tree = SegmentTree(arr, "min")
    
    # Test range queries
    assert tree.query(0, 5) == 1  # Min of entire array
    assert tree.query(0, 2) == 2  # Min of [5, 2, 8]
    assert tree.query(3, 5) == 1  # Min of [1, 9, 3]
    
    # Test updates
    tree.update(3, 7)  # Change 1 to 7
    assert tree.query(0, 5) == 2  # New min after update
    assert tree.query(3, 5) == 3  # New min of subrange after update

def test_segment_tree_max():
    """Test segment tree with maximum operation."""
    # Basic test case
    arr = [5, 2, 8, 1, 9, 3]
    tree = SegmentTree(arr, "max")
    
    # Test range queries
    assert tree.query(0, 5) == 9  # Max of entire array
    assert tree.query(0, 2) == 8  # Max of [5, 2, 8]
    assert tree.query(3, 5) == 9  # Max of [1, 9, 3]
    
    # Test updates
    tree.update(4, 4)  # Change 9 to 4
    assert tree.query(0, 5) == 8  # New max after update
    assert tree.query(3, 5) == 4  # New max of subrange after update

def test_empty_array():
    """Test segment tree with an empty array."""
    arr = []
    tree = SegmentTree(arr, "sum")
    
    # No valid queries on an empty array
    with pytest.raises(IndexError):
        tree.query(0, 0)

def test_invalid_operation():
    """Test segment tree with an invalid operation."""
    arr = [1, 2, 3]
    with pytest.raises(ValueError):
        SegmentTree(arr, "invalid")

def test_invalid_query_range():
    """Test segment tree with invalid query range."""
    arr = [1, 2, 3, 4, 5]
    tree = SegmentTree(arr, "sum")
    
    # Out of bounds
    with pytest.raises(IndexError):
        tree.query(-1, 3)
    with pytest.raises(IndexError):
        tree.query(1, 5)
    # Left > Right
    with pytest.raises(IndexError):
        tree.query(3, 2)

def test_invalid_update_index():
    """Test segment tree with invalid update index."""
    arr = [1, 2, 3, 4, 5]
    tree = SegmentTree(arr, "sum")
    
    with pytest.raises(IndexError):
        tree.update(-1, 10)
    with pytest.raises(IndexError):
        tree.update(5, 10)
