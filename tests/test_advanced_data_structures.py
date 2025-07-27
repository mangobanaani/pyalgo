"""
Tests for Advanced Data Structures

Test suite for advanced data structures including trees, heaps, 
union-find, and persistent data structures.
"""

import unittest
import random
from advanced_data_structures import (
    AVLTree, RedBlackTree, SplayTree,
    BinaryHeap, MinHeap, MaxHeap, FibonacciHeap, BinomialHeap,
    UnionFind, WeightedUnionFind,
    FenwickTree, FenwickTree2D,
    PersistentArray, PersistentVector
)


class TestAVLTree(unittest.TestCase):
    """Test cases for AVL Tree."""
    
    def setUp(self):
        self.avl = AVLTree()
    
    def test_insertion_and_search(self):
        """Test basic insertion and search operations."""
        keys = [10, 5, 15, 3, 7, 12, 18]
        for key in keys:
            self.avl.insert(key, f"value_{key}")
        
        for key in keys:
            self.assertEqual(self.avl.search(key), f"value_{key}")
        
        self.assertIsNone(self.avl.search(100))
    
    def test_deletion(self):
        """Test deletion operations."""
        keys = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8]
        for key in keys:
            self.avl.insert(key, f"value_{key}")
        
        # Test deletion of leaf, node with one child, and node with two children
        self.assertTrue(self.avl.delete(1))   # Leaf
        self.assertTrue(self.avl.delete(15))  # Node with two children
        self.assertTrue(self.avl.delete(5))   # Node with one child
        
        self.assertIsNone(self.avl.search(1))
        self.assertIsNone(self.avl.search(15))
        self.assertIsNone(self.avl.search(5))
        
        self.assertFalse(self.avl.delete(100))  # Non-existent key
    
    def test_balance_property(self):
        """Test that AVL tree maintains balance property."""
        # Insert in sorted order (worst case for unbalanced tree)
        for i in range(1, 16):
            self.avl.insert(i, f"value_{i}")
        
        self.assertTrue(self.avl.is_balanced())
        self.assertLessEqual(self.avl.height(), 5)  # Height should be O(log n)
    
    def test_traversals(self):
        """Test tree traversal methods."""
        keys = [10, 5, 15, 3, 7, 12, 18]
        for key in keys:
            self.avl.insert(key, f"value_{key}")
        
        inorder = self.avl.inorder_traversal()
        self.assertEqual(inorder, [3, 5, 7, 10, 12, 15, 18])
        
        preorder = self.avl.preorder_traversal()
        self.assertTrue(len(preorder) == len(keys))
        self.assertEqual(preorder[0], 10)  # Root should be first


class TestBinaryHeap(unittest.TestCase):
    """Test cases for Binary Heap."""
    
    def test_min_heap(self):
        """Test min heap operations."""
        heap = MinHeap()
        values = [5, 3, 8, 1, 9, 2, 7]
        
        for val in values:
            heap.insert(val)
        
        # Extract all elements - should come out in sorted order
        result = []
        while not heap.is_empty():
            result.append(heap.extract_min())
        
        self.assertEqual(result, sorted(values))
    
    def test_max_heap(self):
        """Test max heap operations."""
        heap = MaxHeap()
        values = [5, 3, 8, 1, 9, 2, 7]
        
        for val in values:
            heap.insert(val)
        
        # Extract all elements - should come out in reverse sorted order
        result = []
        while not heap.is_empty():
            result.append(heap.extract_max())
        
        self.assertEqual(result, sorted(values, reverse=True))
    
    def test_heapify(self):
        """Test heap construction from array."""
        values = [5, 3, 8, 1, 9, 2, 7]
        heap = BinaryHeap.heapify(values, is_min_heap=True)
        
        result = []
        while not heap.is_empty():
            result.append(heap.extract())
        
        self.assertEqual(result, sorted(values))
    
    def test_peek(self):
        """Test peek operation."""
        heap = MinHeap()
        self.assertRaises(IndexError, heap.peek)
        
        heap.insert(5)
        heap.insert(3)
        heap.insert(8)
        
        self.assertEqual(heap.peek(), 3)
        self.assertEqual(heap.size(), 3)  # Size shouldn't change


class TestFibonacciHeap(unittest.TestCase):
    """Test cases for Fibonacci Heap."""
    
    def test_basic_operations(self):
        """Test basic heap operations."""
        heap = FibonacciHeap()
        
        # Insert nodes
        node1 = heap.insert(5, "data5")
        node2 = heap.insert(3, "data3")
        node3 = heap.insert(8, "data8")
        node4 = heap.insert(1, "data1")
        
        # Test find min
        min_node = heap.find_min()
        self.assertEqual(min_node.key, 1)
        self.assertEqual(min_node.data, "data1")
        
        # Test extract min
        extracted = heap.extract_min()
        self.assertEqual(extracted.key, 1)
        
        # Test decrease key
        heap.decrease_key(node3, 2)
        min_node = heap.find_min()
        self.assertEqual(min_node.key, 2)
    
    def test_merge(self):
        """Test heap merge operation."""
        heap1 = FibonacciHeap()
        heap2 = FibonacciHeap()
        
        heap1.insert(5, "h1_5")
        heap1.insert(3, "h1_3")
        
        heap2.insert(8, "h2_8")
        heap2.insert(1, "h2_1")
        
        merged = heap1.merge(heap2)
        self.assertEqual(merged.size(), 4)
        
        min_node = merged.find_min()
        self.assertEqual(min_node.key, 1)


class TestUnionFind(unittest.TestCase):
    """Test cases for Union-Find data structure."""
    
    def test_basic_operations(self):
        """Test basic union and find operations."""
        uf = UnionFind(10)
        
        # Initially, each element is its own component
        self.assertEqual(uf.component_count(), 10)
        for i in range(10):
            self.assertEqual(uf.component_size(i), 1)
        
        # Union some elements
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(4, 5)
        
        # Test connectivity
        self.assertTrue(uf.connected(1, 3))
        self.assertTrue(uf.connected(4, 5))
        self.assertFalse(uf.connected(1, 4))
        
        # Test component sizes
        self.assertEqual(uf.component_size(1), 3)
        self.assertEqual(uf.component_size(4), 2)
        self.assertEqual(uf.component_count(), 7)
    
    def test_weighted_union_find(self):
        """Test weighted union-find operations."""
        wuf = WeightedUnionFind(5)
        
        # Set some weights
        wuf.union(0, 1, 5)    # weight[1] - weight[0] = 5
        wuf.union(1, 2, 3)    # weight[2] - weight[1] = 3
        wuf.union(3, 4, 2)    # weight[4] - weight[3] = 2
        
        # Test relative weights
        self.assertEqual(wuf.diff(0, 2), 8)  # 5 + 3 = 8
        self.assertFalse(wuf.connected(0, 3))


class TestFenwickTree(unittest.TestCase):
    """Test cases for Fenwick Tree (Binary Indexed Tree)."""
    
    def test_range_sum_queries(self):
        """Test range sum queries and updates."""
        arr = [1, 3, 5, 7, 9, 11]
        ft = FenwickTree(arr)
        
        # Test prefix sums
        self.assertEqual(ft.prefix_sum(0), 1)
        self.assertEqual(ft.prefix_sum(2), 9)   # 1 + 3 + 5
        self.assertEqual(ft.prefix_sum(5), 36)  # Sum of all elements
        
        # Test range sums
        self.assertEqual(ft.range_sum(1, 3), 15)  # 3 + 5 + 7
        self.assertEqual(ft.range_sum(2, 4), 21)  # 5 + 7 + 9
        
        # Test updates
        ft.update(2, 10)  # Change arr[2] from 5 to 15
        self.assertEqual(ft.range_sum(1, 3), 25)  # 3 + 15 + 7
    
    def test_2d_fenwick_tree(self):
        """Test 2D Fenwick Tree operations."""
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        ft2d = FenwickTree2D(matrix)
        
        # Test rectangle sum
        total_sum = ft2d.range_sum(0, 0, 2, 2)
        self.assertEqual(total_sum, 45)  # Sum of all elements
        
        # Test partial rectangle
        partial_sum = ft2d.range_sum(0, 0, 1, 1)
        self.assertEqual(partial_sum, 12)  # 1 + 2 + 4 + 5
        
        # Test update
        ft2d.update(1, 1, 10)  # Add 10 to matrix[1][1]
        new_sum = ft2d.range_sum(0, 0, 2, 2)
        self.assertEqual(new_sum, 55)


class TestPersistentArray(unittest.TestCase):
    """Test cases for Persistent Array."""
    
    def test_basic_operations(self):
        """Test basic persistent array operations."""
        # Create initial array
        arr1 = PersistentArray([1, 2, 3, 4, 5])
        
        # Test access
        self.assertEqual(arr1.get(0), 1)
        self.assertEqual(arr1.get(4), 5)
        
        # Test immutability - set creates new version
        arr2 = arr1.set(2, 99)
        self.assertEqual(arr1.get(2), 3)   # Original unchanged
        self.assertEqual(arr2.get(2), 99)  # New version changed
        
        # Test append
        arr3 = arr2.append(6)
        self.assertEqual(arr2.size(), 5)
        self.assertEqual(arr3.size(), 6)
        self.assertEqual(arr3.get(5), 6)
    
    def test_slicing(self):
        """Test array slicing operations."""
        arr = PersistentArray(list(range(10)))
        
        # Test slice
        sub_arr = arr.slice(2, 7)
        expected = [2, 3, 4, 5, 6]
        self.assertEqual(sub_arr.to_list(), expected)
        
        # Test concatenation
        arr1 = PersistentArray([1, 2, 3])
        arr2 = PersistentArray([4, 5, 6])
        combined = arr1.concat(arr2)
        self.assertEqual(combined.to_list(), [1, 2, 3, 4, 5, 6])
    
    def test_persistent_vector(self):
        """Test PersistentVector operations."""
        vec = PersistentVector([1, 2, 3])
        
        # Test push
        vec2 = vec.push(4)
        self.assertEqual(vec.size(), 3)
        self.assertEqual(vec2.size(), 4)
        
        # Test peek
        self.assertEqual(vec2.peek(), 4)
        
        # Test pop
        vec3, popped = vec2.pop_back()
        self.assertEqual(popped, 4)
        self.assertEqual(vec3.size(), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple data structures."""
    
    def test_graph_connectivity_with_union_find(self):
        """Test graph connectivity using Union-Find."""
        # Create a graph with 6 vertices
        uf = UnionFind(6)
        
        # Add edges (representing connections)
        edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
        for u, v in edges:
            uf.union(u, v)
        
        # Test connectivity
        self.assertTrue(uf.connected(0, 2))   # Connected through 1
        self.assertTrue(uf.connected(3, 5))   # Connected through 4
        self.assertFalse(uf.connected(0, 3))  # Different components
        
        # Should have 2 components: {0,1,2} and {3,4,5}
        self.assertEqual(uf.component_count(), 2)
    
    def test_priority_queue_with_heaps(self):
        """Test priority queue operations using different heaps."""
        from advanced_data_structures import PriorityQueue, FibonacciPriorityQueue
        
        # Test with binary heap
        pq = PriorityQueue()
        tasks = [("task_high", 1), ("task_low", 5), ("task_medium", 3)]
        
        for task, priority in tasks:
            pq.put(task, priority)
        
        # Should come out in priority order (lower number = higher priority)
        self.assertEqual(pq.get(), "task_high")
        self.assertEqual(pq.get(), "task_medium")
        self.assertEqual(pq.get(), "task_low")
        
        # Test with Fibonacci heap
        fpq = FibonacciPriorityQueue()
        for task, priority in tasks:
            fpq.put(task, priority)
        
        # Test decrease priority operation
        fpq.decrease_priority("task_low", 0)  # Make it highest priority
        self.assertEqual(fpq.get(), "task_low")


class TestPerformance(unittest.TestCase):
    """Performance tests for data structures."""
    
    def test_avl_tree_performance(self):
        """Test AVL tree performance with large dataset."""
        avl = AVLTree()
        n = 1000
        
        # Insert random values
        values = list(range(n))
        random.shuffle(values)
        
        for val in values:
            avl.insert(val, f"value_{val}")
        
        # Tree should remain balanced
        self.assertTrue(avl.is_balanced())
        self.assertLessEqual(avl.height(), 15)  # Should be ~log2(1000) ≈ 10
        
        # All values should be searchable
        for val in values[:100]:  # Test subset for speed
            self.assertEqual(avl.search(val), f"value_{val}")
    
    def test_fenwick_tree_performance(self):
        """Test Fenwick tree performance with many operations."""
        n = 1000
        arr = [1] * n
        ft = FenwickTree(arr)
        
        # Test many updates and queries
        for i in range(100):
            idx = random.randint(0, n-1)
            val = random.randint(1, 10)
            ft.update(idx, val)
            
            # Query random range
            left = random.randint(0, n-2)
            right = random.randint(left, n-1)
            result = ft.range_sum(left, right)
            self.assertIsInstance(result, (int, float))


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestAVLTree, TestBinaryHeap, TestFibonacciHeap,
        TestUnionFind, TestFenwickTree, TestPersistentArray,
        TestIntegration, TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nRan {result.testsRun} tests")
    if result.failures:
        print(f"FAILURES: {len(result.failures)}")
    if result.errors:
        print(f"ERRORS: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed.")
