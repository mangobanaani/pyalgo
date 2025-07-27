class MergeSort:
    @staticmethod
    def merge(left, right, arr):
        """
        Merge two sorted subarrays into the original array
        """
        i = j = 0
        while i + j < len(arr):
            if j == len(right) or (i < len(left) and left[i] <= right[j]):
                arr[i + j] = left[i]
                i += 1
            else:
                arr[i + j] = right[j]
                j += 1

    @staticmethod
    def merge_sort(arr):
        """
        Merge Sort implementation
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(arr)
        if n < 2:
            return
        
        mid = n // 2  # Split point
        left = arr[0:mid]     # Left half
        right = arr[mid:n]    # Right half
        
        # Recursively sort both halves
        MergeSort.merge_sort(left)
        MergeSort.merge_sort(right)
        
        # Merge the sorted halves
        MergeSort.merge(left, right, arr)
