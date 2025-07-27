import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from searching.binary_search import BinarySearch
from searching.linear_search import LinearSearch


class TestSearchingAlgorithms(unittest.TestCase):
    """Test suite for searching algorithms"""
    
    def setUp(self):
        """Set up test cases"""
        self.sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        self.unsorted_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    
    def test_binary_search(self):
        """Test binary search"""
        # Test existing elements
        self.assertEqual(BinarySearch.binary_search(self.sorted_array, 7), 3)
        self.assertEqual(BinarySearch.binary_search(self.sorted_array, 1), 0)
        self.assertEqual(BinarySearch.binary_search(self.sorted_array, 19), 9)
        
        # Test non-existing elements
        self.assertEqual(BinarySearch.binary_search(self.sorted_array, 0), -1)
        self.assertEqual(BinarySearch.binary_search(self.sorted_array, 20), -1)
        self.assertEqual(BinarySearch.binary_search(self.sorted_array, 8), -1)
        
        # Test empty array
        self.assertEqual(BinarySearch.binary_search([], 5), -1)
    
    def test_binary_search_recursive(self):
        """Test recursive binary search"""
        # Test existing elements
        self.assertEqual(BinarySearch.binary_search_recursive(self.sorted_array, 7), 3)
        self.assertEqual(BinarySearch.binary_search_recursive(self.sorted_array, 1), 0)
        self.assertEqual(BinarySearch.binary_search_recursive(self.sorted_array, 19), 9)
        
        # Test non-existing elements
        self.assertEqual(BinarySearch.binary_search_recursive(self.sorted_array, 0), -1)
        self.assertEqual(BinarySearch.binary_search_recursive(self.sorted_array, 20), -1)
    
    def test_linear_search(self):
        """Test linear search"""
        # Test with sorted array
        self.assertEqual(LinearSearch.linear_search(self.sorted_array, 7), 3)
        self.assertEqual(LinearSearch.linear_search(self.sorted_array, 1), 0)
        
        # Test with unsorted array
        self.assertEqual(LinearSearch.linear_search(self.unsorted_array, 3), 0)
        self.assertEqual(LinearSearch.linear_search(self.unsorted_array, 1), 1)
        self.assertEqual(LinearSearch.linear_search(self.unsorted_array, 5), 4)
        
        # Test non-existing elements
        self.assertEqual(LinearSearch.linear_search(self.unsorted_array, 99), -1)
    
    def test_linear_search_all_occurrences(self):
        """Test finding all occurrences"""
        indices = LinearSearch.linear_search_all_occurrences(self.unsorted_array, 5)
        self.assertEqual(indices, [4, 8, 10])
        
        indices = LinearSearch.linear_search_all_occurrences(self.unsorted_array, 1)
        self.assertEqual(indices, [1, 3])
        
        indices = LinearSearch.linear_search_all_occurrences(self.unsorted_array, 99)
        self.assertEqual(indices, [])
    
    def test_find_max_min(self):
        """Test finding maximum and minimum elements"""
        max_index, max_val = LinearSearch.find_max(self.unsorted_array)
        self.assertEqual(max_val, 9)
        self.assertEqual(max_index, 5)
        
        min_index, min_val = LinearSearch.find_min(self.unsorted_array)
        self.assertEqual(min_val, 1)
        self.assertEqual(min_index, 1)  # First occurrence
    
    def test_linear_search_with_condition(self):
        """Test linear search with custom condition"""
        # Find first even number
        index = LinearSearch.linear_search_with_condition(
            self.unsorted_array, 
            lambda x: x % 2 == 0
        )
        self.assertEqual(index, 2)  # First even number is 4 at index 2
        
        # Find first number greater than 8
        index = LinearSearch.linear_search_with_condition(
            self.unsorted_array, 
            lambda x: x > 8
        )
        self.assertEqual(index, 5)  # First number > 8 is 9 at index 5


if __name__ == "__main__":
    unittest.main()
