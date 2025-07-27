#!/usr/bin/env python3
"""
PyAlgo Demo Script
Demonstrates the usage of various algorithms and data structures.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sorting import MergeSort, QuickSort, BubbleSort, HeapSort
from data_structures import Stack, Queue, LinkedList, BinarySearchTree
from searching import BinarySearch, LinearSearch
from graph import Graph, GraphAlgorithms
from dynamic_programming import DynamicProgramming


def demo_sorting():
    """Demonstrate sorting algorithms"""
    print("=" * 50)
    print("SORTING ALGORITHMS DEMO")
    print("=" * 50)
    
    test_array = [64, 34, 25, 12, 22, 11, 90, 5]
    print(f"Original array: {test_array}")
    
    # Merge Sort
    arr = test_array.copy()
    MergeSort.merge_sort(arr)
    print(f"Merge Sort:     {arr}")
    
    # Quick Sort
    arr = test_array.copy()
    QuickSort.quick_sort(arr)
    print(f"Quick Sort:     {arr}")
    
    # Bubble Sort
    arr = test_array.copy()
    BubbleSort.bubble_sort(arr)
    print(f"Bubble Sort:    {arr}")
    
    # Heap Sort
    arr = test_array.copy()
    HeapSort.heap_sort(arr)
    print(f"Heap Sort:      {arr}")


def demo_data_structures():
    """Demonstrate data structures"""
    print("\n" + "=" * 50)
    print("DATA STRUCTURES DEMO")
    print("=" * 50)
    
    # Stack Demo
    print("\n--- Stack Demo ---")
    stack = Stack()
    for i in [1, 2, 3, 4, 5]:
        stack.push(i)
        print(f"Pushed {i}, stack size: {stack.len()}")
    
    print("Popping elements:")
    while not stack.is_empty():
        print(f"Popped: {stack.pop()}")
    
    # Queue Demo
    print("\n--- Queue Demo ---")
    queue = Queue()
    for i in [1, 2, 3, 4, 5]:
        queue.enqueue(i)
        print(f"Enqueued {i}, queue size: {queue.len()}")
    
    print("Dequeuing elements:")
    while not queue.is_empty():
        print(f"Dequeued: {queue.dequeue()}")
    
    # Linked List Demo
    print("\n--- Linked List Demo ---")
    ll = LinkedList()
    for i in [1, 2, 3, 4, 5]:
        ll.append(i)
    print(f"List after appending 1-5: {ll.display()}")
    
    ll.prepend(0)
    print(f"After prepending 0: {ll.display()}")
    
    ll.delete(3)
    print(f"After deleting 3: {ll.display()}")
    
    ll.reverse()
    print(f"After reversing: {ll.display()}")
    
    # Binary Search Tree Demo
    print("\n--- Binary Search Tree Demo ---")
    bst = BinarySearchTree()
    values = [50, 30, 70, 20, 40, 60, 80]
    for val in values:
        bst.insert(val)
    
    print(f"Inserted values: {values}")
    print(f"Inorder traversal: {bst.inorder_traversal()}")
    print(f"Preorder traversal: {bst.preorder_traversal()}")
    print(f"Search for 40: {bst.search(40)}")
    print(f"Search for 99: {bst.search(99)}")


def demo_searching():
    """Demonstrate searching algorithms"""
    print("\n" + "=" * 50)
    print("SEARCHING ALGORITHMS DEMO")
    print("=" * 50)
    
    sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    target = 13
    
    print(f"Sorted array: {sorted_array}")
    print(f"Searching for: {target}")
    
    # Binary Search
    index = BinarySearch.binary_search(sorted_array, target)
    print(f"Binary Search: Found at index {index}")
    
    # Linear Search
    index = LinearSearch.linear_search(sorted_array, target)
    print(f"Linear Search: Found at index {index}")
    
    # Search for all occurrences
    array_with_duplicates = [1, 3, 5, 5, 5, 7, 9, 9, 11]
    indices = LinearSearch.linear_search_all_occurrences(array_with_duplicates, 5)
    print(f"All occurrences of 5 in {array_with_duplicates}: {indices}")
    
    # Find max and min
    test_array = [64, 34, 25, 12, 22, 11, 90, 5]
    max_idx, max_val = LinearSearch.find_max(test_array)
    min_idx, min_val = LinearSearch.find_min(test_array)
    print(f"In array {test_array}:")
    print(f"Maximum: {max_val} at index {max_idx}")
    print(f"Minimum: {min_val} at index {min_idx}")


def demo_graph_algorithms():
    """Demonstrate graph algorithms"""
    print("\n" + "=" * 50)
    print("GRAPH ALGORITHMS DEMO")
    print("=" * 50)
    
    # Create a sample graph
    graph = Graph()
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('C', 'E')]
    
    for u, v in edges:
        graph.add_edge(u, v)
    
    print("Graph edges:", edges)
    
    # BFS traversal
    bfs_result = GraphAlgorithms.bfs(graph, 'A')
    print(f"BFS from A: {bfs_result}")
    
    # DFS traversal
    dfs_result = GraphAlgorithms.dfs(graph, 'A')
    print(f"DFS from A: {dfs_result}")
    
    # Check connectivity
    has_path = GraphAlgorithms.has_path(graph, 'A', 'E')
    print(f"Path from A to E exists: {has_path}")
    
    # Find shortest path
    shortest_path = GraphAlgorithms.find_shortest_path(graph, 'A', 'E')
    print(f"Shortest path from A to E: {shortest_path}")


def demo_dynamic_programming():
    """Demonstrate dynamic programming algorithms"""
    print("\n" + "=" * 50)
    print("DYNAMIC PROGRAMMING DEMO")
    print("=" * 50)
    
    # Fibonacci
    n = 10
    fib_memo = DynamicProgramming.fibonacci_memo(n)
    fib_tab = DynamicProgramming.fibonacci_tabulation(n)
    fib_opt = DynamicProgramming.fibonacci_optimized(n)
    print(f"Fibonacci({n}):")
    print(f"  Memoization: {fib_memo}")
    print(f"  Tabulation:  {fib_tab}")
    print(f"  Optimized:   {fib_opt}")
    
    # Longest Common Subsequence
    text1, text2 = "ABCDGH", "AEDFHR"
    lcs_length = DynamicProgramming.longest_common_subsequence(text1, text2)
    print(f"\nLongest Common Subsequence:")
    print(f"  Text 1: {text1}")
    print(f"  Text 2: {text2}")
    print(f"  LCS Length: {lcs_length}")
    
    # 0/1 Knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value = DynamicProgramming.knapsack_01(weights, values, capacity)
    print(f"\n0/1 Knapsack:")
    print(f"  Weights: {weights}")
    print(f"  Values:  {values}")
    print(f"  Capacity: {capacity}")
    print(f"  Maximum value: {max_value}")
    
    # Coin Change
    coins = [1, 3, 4]
    amount = 6
    min_coins = DynamicProgramming.coin_change(coins, amount)
    print(f"\nCoin Change:")
    print(f"  Coins: {coins}")
    print(f"  Amount: {amount}")
    print(f"  Minimum coins needed: {min_coins}")
    
    # Edit Distance
    word1, word2 = "kitten", "sitting"
    distance = DynamicProgramming.edit_distance(word1, word2)
    print(f"\nEdit Distance:")
    print(f"  Word 1: {word1}")
    print(f"  Word 2: {word2}")
    print(f"  Edit distance: {distance}")


def main():
    """Run all demonstrations"""
    print("PyAlgo - Python Algorithms and Data Structures")
    print("Comprehensive demonstration of implemented algorithms")
    
    demo_sorting()
    demo_data_structures()
    demo_searching()
    demo_graph_algorithms()
    demo_dynamic_programming()
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETED")
    print("=" * 50)
    print("Check the tests/ directory for comprehensive test suites.")
    print("Run 'python -m pytest tests/ -v' to execute all tests.")


if __name__ == "__main__":
    main()
