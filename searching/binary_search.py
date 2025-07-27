class BinarySearch:
    @staticmethod
    def binary_search(arr, target):
        """
        Binary search implementation for sorted arrays.
        Time Complexity: O(log n)
        Space Complexity: O(1)
        
        Args:
            arr: Sorted array to search in
            target: Value to search for
            
        Returns:
            Index of target if found, -1 otherwise
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = left + (right - left) // 2  # Prevent overflow
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
    
    @staticmethod
    def binary_search_recursive(arr, target, left=None, right=None):
        """
        Recursive binary search implementation.
        Time Complexity: O(log n)
        Space Complexity: O(log n) due to recursion
        """
        if left is None:
            left = 0
        if right is None:
            right = len(arr) - 1
        
        if left > right:
            return -1
        
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return BinarySearch.binary_search_recursive(arr, target, mid + 1, right)
        else:
            return BinarySearch.binary_search_recursive(arr, target, left, mid - 1)
    
    @staticmethod
    def binary_search_leftmost(arr, target):
        """
        Find the leftmost occurrence of target in a sorted array.
        Returns the index where target should be inserted to keep array sorted.
        """
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    @staticmethod
    def binary_search_rightmost(arr, target):
        """
        Find the rightmost occurrence of target in a sorted array.
        Returns the index after the last occurrence of target.
        """
        left, right = 0, len(arr)
        
        while left < right:
            mid = left + (right - left) // 2
            
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        
        return left
