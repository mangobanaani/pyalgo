class LinearSearch:
    @staticmethod
    def linear_search(arr, target):
        """
        Linear search implementation.
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            arr: Array to search in (can be unsorted)
            target: Value to search for
            
        Returns:
            Index of target if found, -1 otherwise
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1
    
    @staticmethod
    def linear_search_all_occurrences(arr, target):
        """
        Find all occurrences of target in the array.
        
        Returns:
            List of indices where target is found
        """
        indices = []
        for i in range(len(arr)):
            if arr[i] == target:
                indices.append(i)
        return indices
    
    @staticmethod
    def linear_search_with_condition(arr, condition_func):
        """
        Linear search with a custom condition function.
        
        Args:
            arr: Array to search in
            condition_func: Function that returns True for matching elements
            
        Returns:
            Index of first element that satisfies condition, -1 otherwise
        """
        for i in range(len(arr)):
            if condition_func(arr[i]):
                return i
        return -1
    
    @staticmethod
    def find_max(arr):
        """Find the maximum element and its index"""
        if not arr:
            raise ValueError("Array is empty")
        
        max_val = arr[0]
        max_index = 0
        
        for i in range(1, len(arr)):
            if arr[i] > max_val:
                max_val = arr[i]
                max_index = i
        
        return max_index, max_val
    
    @staticmethod
    def find_min(arr):
        """Find the minimum element and its index"""
        if not arr:
            raise ValueError("Array is empty")
        
        min_val = arr[0]
        min_index = 0
        
        for i in range(1, len(arr)):
            if arr[i] < min_val:
                min_val = arr[i]
                min_index = i
        
        return min_index, min_val
