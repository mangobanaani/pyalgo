# Searching Algorithms

Efficient algorithms for finding elements in data structures with different requirements and performance characteristics.

## Algorithms Included

### Binary Search
- **File**: `binary_search.py`
- **Time Complexity**: O(log n)
- **Space Complexity**: O(1) iterative, O(log n) recursive
- **Requirement**: Array must be sorted
- **Description**: Efficiently searches a sorted array by repeatedly dividing the search interval in half
- **Variants**: Standard search, leftmost occurrence, rightmost occurrence
- **Best Use**: Searching in large sorted datasets

### Linear Search
- **File**: `linear_search.py`
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Requirement**: None (works on unsorted arrays)
- **Description**: Simple sequential search that checks each element until target is found
- **Features**: Find all occurrences, custom comparison conditions
- **Best Use**: Small datasets or unsorted data

## Usage Examples

```python
from searching.binary_search import BinarySearch
from searching.linear_search import LinearSearch

# Sorted array for binary search
sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# Binary search
index = BinarySearch.binary_search(sorted_arr, 7)
print(f"Binary search found 7 at index: {index}")

# Find leftmost occurrence
left_index = BinarySearch.binary_search_leftmost(sorted_arr, 7)
print(f"Leftmost occurrence at index: {left_index}")

# Unsorted array for linear search
unsorted_arr = [3, 1, 4, 1, 5, 9, 2, 6]

# Linear search
linear_index = LinearSearch.linear_search(unsorted_arr, 5)
print(f"Linear search found 5 at index: {linear_index}")

# Find all occurrences
all_indices = LinearSearch.linear_search_all(unsorted_arr, 1)
print(f"All occurrences of 1: {all_indices}")
```

## Performance Comparison

| Algorithm | Time Complexity | Space Complexity | Requires Sorted | Best For |
|-----------|----------------|------------------|----------------|----------|
| Binary Search | O(log n) | O(1) | Yes | Large sorted datasets |
| Linear Search | O(n) | O(1) | No | Small or unsorted data |

## Performance Analysis

For a sorted array of size n:
- **Binary Search**: Maximum of log₂(n) comparisons
- **Linear Search**: Maximum of n comparisons

### Example Performance Difference
- Array size 1,000: Binary search max 10 steps vs Linear search max 1,000 steps
- Array size 1,000,000: Binary search max 20 steps vs Linear search max 1,000,000 steps

## When to Use Which Algorithm

- **Binary Search**: 
  - Data is already sorted
  - Large datasets (> 100 elements)
  - Frequent search operations
  - Memory efficiency is important

- **Linear Search**:
  - Data is unsorted and sorting is expensive
  - Small datasets (< 100 elements)
  - Need to find all occurrences
  - Custom comparison logic is required
