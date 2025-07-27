"""
Randomized Quicksort and Selection algorithms.

This module implements randomized versions of quicksort and the selection algorithm
that provide good average-case performance and avoid worst-case scenarios.
"""

import random
from typing import List, TypeVar

T = TypeVar('T')


def randomized_quicksort(arr: List[T]) -> List[T]:
    """
    Randomized quicksort algorithm.
    
    Uses random pivot selection to achieve O(n log n) expected time complexity
    and avoid worst-case O(n²) performance on sorted arrays.
    
    Args:
        arr: List to sort
        
    Returns:
        New sorted list
        
    Time Complexity: O(n log n) expected, O(n²) worst case
    Space Complexity: O(log n) expected for recursion stack
    """
    if len(arr) <= 1:
        return arr
    
    # Choose random pivot
    pivot_idx = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_idx]
    
    # Partition around pivot
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort and combine
    return randomized_quicksort(left) + middle + randomized_quicksort(right)


def randomized_select(arr: List[T], k: int) -> T:
    """
    Randomized selection algorithm to find the k-th smallest element.
    
    Uses randomized partitioning similar to quicksort to find the k-th
    order statistic in expected O(n) time.
    
    Args:
        arr: List to select from (will be modified)
        k: Index of element to find (0-based)
        
    Returns:
        The k-th smallest element
        
    Raises:
        IndexError: If k is out of bounds
        
    Time Complexity: O(n) expected, O(n²) worst case
    Space Complexity: O(log n) expected for recursion
    """
    if not 0 <= k < len(arr):
        raise IndexError(f"Index {k} out of bounds for array of length {len(arr)}")
    
    return _randomized_select_helper(arr, 0, len(arr) - 1, k)


def _randomized_select_helper(arr: List[T], left: int, right: int, k: int) -> T:
    """Helper function for randomized select."""
    if left == right:
        return arr[left]
    
    # Randomized partition
    pivot_idx = _randomized_partition(arr, left, right)
    
    if k == pivot_idx:
        return arr[k]
    elif k < pivot_idx:
        return _randomized_select_helper(arr, left, pivot_idx - 1, k)
    else:
        return _randomized_select_helper(arr, pivot_idx + 1, right, k)


def _randomized_partition(arr: List[T], left: int, right: int) -> int:
    """
    Randomized partition function.
    
    Chooses a random pivot and partitions the array around it.
    Returns the final position of the pivot.
    """
    # Choose random pivot and swap with last element
    pivot_idx = random.randint(left, right)
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
    
    # Standard partition with pivot at end
    pivot = arr[right]
    i = left - 1
    
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


def quicksort_inplace(arr: List[T], left: int = None, right: int = None) -> None:
    """
    In-place randomized quicksort.
    
    Sorts the array in place using randomized pivot selection.
    
    Args:
        arr: List to sort in place
        left: Left boundary (default: 0)
        right: Right boundary (default: len(arr) - 1)
        
    Time Complexity: O(n log n) expected, O(n²) worst case
    Space Complexity: O(log n) expected for recursion stack
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    if left < right:
        pivot_idx = _randomized_partition(arr, left, right)
        quicksort_inplace(arr, left, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, right)
