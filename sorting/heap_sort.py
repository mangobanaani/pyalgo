class HeapSort:
    @staticmethod
    def heapify(arr, n, i):
        """
        To heapify a subtree rooted with node i which is an index in arr[].
        n is size of heap
        """
        largest = i  # Initialize largest as root
        left = 2 * i + 1  # left child
        right = 2 * i + 2  # right child
        
        # If left child exists and is greater than root
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        # If right child exists and is greater than largest so far
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        # If largest is not root
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]  # swap
            # Heapify the root
            HeapSort.heapify(arr, n, largest)
    
    @staticmethod
    def heap_sort(arr):
        """
        Main function to do heap sort.
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        n = len(arr)
        
        # Build a maxheap
        # Start from the last non-leaf node and go backwards
        for i in range(n // 2 - 1, -1, -1):
            HeapSort.heapify(arr, n, i)
        
        # One by one extract elements
        for i in range(n - 1, 0, -1):
            # Move current root to end
            arr[0], arr[i] = arr[i], arr[0]
            # Call heapify on the reduced heap
            HeapSort.heapify(arr, i, 0)
