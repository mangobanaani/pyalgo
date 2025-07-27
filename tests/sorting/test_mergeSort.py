import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sorting.merge_sort import MergeSort


class TestMergeSort(unittest.TestCase):
    def test_merge_sort(self):
        """Test basic merge sort functionality"""
        arr = [3, 2, 1]
        MergeSort.merge_sort(arr)
        expected = [1, 2, 3]
        self.assertListEqual(arr, expected, "sort not as expected")
    
    def test_merge_sort_empty(self):
        """Test merge sort with empty array"""
        arr = []
        MergeSort.merge_sort(arr)
        self.assertListEqual(arr, [])
    
    def test_merge_sort_single_element(self):
        """Test merge sort with single element"""
        arr = [42]
        MergeSort.merge_sort(arr)
        self.assertListEqual(arr, [42])
    
    def test_merge_sort_already_sorted(self):
        """Test merge sort with already sorted array"""
        arr = [1, 2, 3, 4, 5]
        MergeSort.merge_sort(arr)
        self.assertListEqual(arr, [1, 2, 3, 4, 5])
    
    def test_merge_sort_reverse_sorted(self):
        """Test merge sort with reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        MergeSort.merge_sort(arr)
        self.assertListEqual(arr, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
