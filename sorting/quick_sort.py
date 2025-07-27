class QuickSort:
    @staticmethod
    def partition(arr, low, high):
        """
        Partition function using Lomuto partition scheme.
        Places the pivot at its correct position and partitions
        around it.
        """
        pivot = arr[high]  # Choose last element as pivot
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    @staticmethod
    def quick_sort_helper(arr, low, high):
        """Helper function for recursive quick sort"""
        if low < high:
            # Partition index
            pi = QuickSort.partition(arr, low, high)
            
            # Recursively sort elements before and after partition
            QuickSort.quick_sort_helper(arr, low, pi - 1)
            QuickSort.quick_sort_helper(arr, pi + 1, high)
    
    @staticmethod
    def quick_sort(arr):
        """
        Main quick sort function.
        Time Complexity: O(n log n) average, O(n²) worst case
        Space Complexity: O(log n) average
        """
        if len(arr) <= 1:
            return
        QuickSort.quick_sort_helper(arr, 0, len(arr) - 1)
