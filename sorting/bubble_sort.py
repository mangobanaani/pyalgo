class BubbleSort:
    @staticmethod
    def bubble_sort(arr):
        """
        Bubble Sort implementation.
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(arr)
        
        # Traverse through all array elements
        for i in range(n):
            # Flag to optimize by checking if any swap occurred
            swapped = False
            
            # Last i elements are already in place
            for j in range(0, n - i - 1):
                # Traverse the array from 0 to n-i-1
                # Swap if the element found is greater than the next element
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            
            # If no swapping occurred, array is sorted
            if not swapped:
                break
    
    @staticmethod
    def bubble_sort_optimized(arr):
        """
        Optimized bubble sort that stops early if array becomes sorted.
        """
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
            if not swapped:
                break
