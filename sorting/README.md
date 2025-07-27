# Sorting Algorithms

A collection of fundamental sorting algorithms with comprehensive analysis and implementations.

## Algorithms Included

### Merge Sort
- **File**: `merge_sort.py`
- **Time Complexity**: O(n log n) in all cases
- **Space Complexity**: O(n)
- **Stability**: Stable
- **Description**: Divide-and-conquer algorithm that splits the array into halves, sorts them recursively, and merges the results
- **Best Use**: When guaranteed O(n log n) performance is required and stability is important

### Quick Sort
- **File**: `quick_sort.py`
- **Time Complexity**: O(n log n) average, O(n²) worst case
- **Space Complexity**: O(log n) average
- **Stability**: Unstable
- **Description**: Divide-and-conquer algorithm that partitions array around a pivot and recursively sorts partitions
- **Best Use**: General-purpose sorting when average-case performance is prioritized

### Heap Sort
- **File**: `heap_sort.py`
- **Time Complexity**: O(n log n) in all cases
- **Space Complexity**: O(1)
- **Stability**: Unstable
- **Description**: Builds a max-heap from the input data, then repeatedly extracts maximum element
- **Best Use**: When memory usage must be minimal and worst-case O(n log n) is required

### Bubble Sort
- **File**: `bubble_sort.py`
- **Time Complexity**: O(n²) average and worst case, O(n) best case
- **Space Complexity**: O(1)
- **Stability**: Stable
- **Description**: Simple algorithm that repeatedly steps through list, compares adjacent elements and swaps them if they're in wrong order
- **Best Use**: Educational purposes and very small datasets

## Usage Examples

```python
from sorting.merge_sort import MergeSort
from sorting.quick_sort import QuickSort

# Example array
arr = [64, 34, 25, 12, 22, 11, 90]

# Merge Sort
merge_sorted = MergeSort.merge_sort(arr.copy())
print(f"Merge Sort: {merge_sorted}")

# Quick Sort
quick_sorted = arr.copy()
QuickSort.quick_sort(quick_sorted)
print(f"Quick Sort: {quick_sorted}")
```

## Performance Comparison

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable |
|-----------|-----------|--------------|------------|-------|---------|
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No |
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes |

## When to Use Which Algorithm

- **Merge Sort**: When you need guaranteed O(n log n) performance and stability
- **Quick Sort**: For general-purpose sorting with good average performance
- **Heap Sort**: When memory is constrained and you need O(n log n) worst-case
- **Bubble Sort**: Only for educational purposes or very small datasets (< 10 elements)
