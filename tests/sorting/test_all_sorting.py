import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sorting.quick_sort import QuickSort
from sorting.bubble_sort import BubbleSort
from sorting.heap_sort import HeapSort


class TestSortingAlgorithms(unittest.TestCase):
    """Test suite for all sorting algorithms"""
    
    def setUp(self):
        """Set up test cases"""
        self.test_cases = [
            [],
            [42],
            [3, 2, 1],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
            [-3, -1, -4, -1, -5, 0, 2],
            [1, 1, 1, 1, 1]
        ]
        self.expected_results = [
            [],
            [42],
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9],
            [-5, -4, -3, -1, -1, 0, 2],
            [1, 1, 1, 1, 1]
        ]
    
    def test_quick_sort(self):
        """Test QuickSort algorithm"""
        for i, test_case in enumerate(self.test_cases):
            with self.subTest(i=i):
                arr = test_case.copy()
                QuickSort.quick_sort(arr)
                self.assertEqual(arr, self.expected_results[i])
    
    def test_bubble_sort(self):
        """Test BubbleSort algorithm"""
        for i, test_case in enumerate(self.test_cases):
            with self.subTest(i=i):
                arr = test_case.copy()
                BubbleSort.bubble_sort(arr)
                self.assertEqual(arr, self.expected_results[i])
    
    def test_bubble_sort_optimized(self):
        """Test optimized BubbleSort algorithm"""
        for i, test_case in enumerate(self.test_cases):
            with self.subTest(i=i):
                arr = test_case.copy()
                BubbleSort.bubble_sort_optimized(arr)
                self.assertEqual(arr, self.expected_results[i])
    
    def test_heap_sort(self):
        """Test HeapSort algorithm"""
        for i, test_case in enumerate(self.test_cases):
            with self.subTest(i=i):
                arr = test_case.copy()
                HeapSort.heap_sort(arr)
                self.assertEqual(arr, self.expected_results[i])


if __name__ == "__main__":
    unittest.main()
