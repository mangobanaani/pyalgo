# Advanced Data Structures

This module contains implementations of sophisticated data structures that provide enhanced performance characteristics and specialized functionality beyond basic collections.

## Overview

Advanced data structures are designed to solve specific computational problems efficiently. They often provide better time complexity for certain operations or enable new types of queries that aren't possible with basic structures.

## Implemented Structures

### Self-Balancing Trees

#### AVL Tree
- **Description**: Height-balanced binary search tree with automatic rebalancing
- **Time Complexity**: O(log n) for search, insert, delete
- **Space Complexity**: O(n)
- **Use Cases**: When you need guaranteed logarithmic performance
- **Key Features**: Automatic balancing, rotation operations, height tracking

#### Red-Black Tree
- **Description**: Self-balancing BST with color-coded nodes following specific rules
- **Time Complexity**: O(log n) for search, insert, delete
- **Space Complexity**: O(n)
- **Use Cases**: Used in many standard library implementations (Java TreeMap, C++ map)
- **Key Features**: Red-black coloring rules, efficient rebalancing

#### Splay Tree
- **Description**: Self-adjusting BST that moves accessed nodes toward root
- **Time Complexity**: O(log n) amortized for all operations
- **Space Complexity**: O(n)
- **Use Cases**: When access patterns have locality of reference
- **Key Features**: Splaying operation, adaptive performance

### Priority Queues and Heaps

#### Binary Heap
- **Description**: Complete binary tree maintaining heap property
- **Time Complexity**: O(log n) insert/extract, O(1) peek
- **Space Complexity**: O(n)
- **Use Cases**: Priority queues, heap sort, graph algorithms
- **Key Features**: Min/max heap variants, heapify operation

#### Fibonacci Heap
- **Description**: Advanced heap with excellent amortized performance
- **Time Complexity**: O(1) amortized insert/decrease-key, O(log n) extract-min
- **Space Complexity**: O(n)
- **Use Cases**: Dijkstra's algorithm, Prim's MST, advanced graph algorithms
- **Key Features**: Lazy consolidation, cascading cuts, efficient merging

#### Binomial Heap
- **Description**: Collection of binomial trees with heap property
- **Time Complexity**: O(log n) for most operations, O(1) merge
- **Space Complexity**: O(n)
- **Use Cases**: When frequent merging of heaps is required
- **Key Features**: Efficient heap merging, binomial tree structure

### Union-Find Structures

#### Disjoint Set Union (Union-Find)
- **Description**: Tracks disjoint sets with efficient union and find operations
- **Time Complexity**: Nearly O(1) amortized with path compression and union by rank
- **Space Complexity**: O(n)
- **Use Cases**: Kruskal's MST, cycle detection, dynamic connectivity
- **Key Features**: Path compression, union by rank, weighted variant

### Range Query Structures

#### Fenwick Tree (Binary Indexed Tree)
- **Description**: Tree structure for efficient prefix sum queries and updates
- **Time Complexity**: O(log n) for query and update
- **Space Complexity**: O(n)
- **Use Cases**: Range sum queries, frequency counting, order statistics
- **Key Features**: 1D and 2D variants, range updates, compact implementation

### Persistent Data Structures

#### Persistent Array
- **Description**: Immutable array with efficient copy-on-write operations
- **Time Complexity**: O(log n) for access and updates
- **Space Complexity**: O(n) with structural sharing
- **Use Cases**: Functional programming, undo operations, version control
- **Key Features**: Immutability, structural sharing, version history

#### Persistent Segment Tree
- **Description**: Immutable segment tree preserving all versions
- **Time Complexity**: O(log n) per operation per version
- **Space Complexity**: O(n log n) for all versions with sharing
- **Use Cases**: Historical queries, competitive programming, temporal analysis
- **Key Features**: Version preservation, structural sharing, range queries

## Usage Examples

### AVL Tree
```python
from advanced_data_structures import AVLTree

avl = AVLTree()
avl.insert(10, "ten")
avl.insert(5, "five")
avl.insert(15, "fifteen")

print(avl.search(10))  # "ten"
print(avl.inorder_traversal())  # [5, 10, 15]
print(avl.is_balanced())  # True
```

### Fibonacci Heap
```python
from advanced_data_structures import FibonacciHeap

heap = FibonacciHeap()
node1 = heap.insert(5, "priority 5")
node2 = heap.insert(3, "priority 3")
node3 = heap.insert(8, "priority 8")

min_node = heap.extract_min()  # node2 (priority 3)
heap.decrease_key(node3, 1)    # Change priority 8 to 1
```

### Union-Find
```python
from advanced_data_structures import UnionFind

uf = UnionFind(10)
uf.union(1, 2)
uf.union(2, 3)

print(uf.find(1) == uf.find(3))  # True (same component)
print(uf.component_size(1))      # 3
print(uf.component_count())      # 8 (10 - 2 unions)
```

### Persistent Array
```python
from advanced_data_structures import PersistentArray

arr1 = PersistentArray([1, 2, 3, 4, 5])
arr2 = arr1.set(2, 99)  # Create new version with arr[2] = 99
arr3 = arr2.append(6)   # Create new version with 6 appended

print(arr1.get(2))  # 3 (original unchanged)
print(arr2.get(2))  # 99
print(arr3.to_list())  # [1, 2, 99, 4, 5, 6]
```

## Performance Characteristics

| Structure | Search | Insert | Delete | Extra Operations |
|-----------|--------|--------|--------|------------------|
| AVL Tree | O(log n) | O(log n) | O(log n) | O(1) height check |
| Red-Black Tree | O(log n) | O(log n) | O(log n) | O(log n) validation |
| Splay Tree | O(log n)* | O(log n)* | O(log n)* | O(log n) split/join |
| Binary Heap | O(n) | O(log n) | O(log n) | O(1) peek |
| Fibonacci Heap | O(n) | O(1)* | O(log n)* | O(1)* decrease-key |
| Binomial Heap | O(log n) | O(log n) | O(log n) | O(log n) merge |
| Union-Find | - | - | - | O(α(n))* union/find |
| Fenwick Tree | - | - | - | O(log n) range query |
| Persistent Array | O(log n) | O(log n) | O(log n) | O(1) version access |

*Amortized time complexity

## Applications

### Competitive Programming
- **Range Queries**: Fenwick Tree, Persistent Segment Tree
- **Dynamic Connectivity**: Union-Find
- **Priority Queues**: Fibonacci Heap for optimal graph algorithms

### System Design
- **Database Indexing**: Red-Black Trees, B-trees
- **Memory Management**: Splay Trees for cache-friendly access
- **Undo Operations**: Persistent data structures

### Algorithm Optimization
- **Graph Algorithms**: Fibonacci Heap for Dijkstra's, Prim's algorithms
- **Dynamic Programming**: Persistent structures for memoization
- **Computational Geometry**: Range trees for multidimensional queries

## Implementation Notes

### Memory Efficiency
- Persistent structures use structural sharing to minimize memory usage
- Union-Find uses path compression to keep tree heights low
- Heaps maintain complete tree structure for cache efficiency

### Thread Safety
- All implementations are not thread-safe by default
- Persistent structures are immutable and inherently safe for concurrent reads
- Mutable structures require external synchronization

### Numerical Stability
- Floating-point keys may cause issues with strict ordering requirements
- Use appropriate comparison functions for custom data types
- Consider precision issues when using with approximate equality

## Best Practices

### Choosing the Right Structure
1. **Frequent Range Queries**: Use Fenwick Tree or Segment Tree
2. **Guaranteed Balance**: Use AVL Tree or Red-Black Tree
3. **Cache-Friendly Access**: Use Splay Tree
4. **Graph Algorithms**: Use Fibonacci Heap for optimal performance
5. **Immutable Requirements**: Use Persistent Array or Persistent Segment Tree
6. **Dynamic Connectivity**: Use Union-Find with optimizations

### Performance Optimization
- Initialize with expected capacity when possible
- Use bulk operations when available
- Consider memory vs. time tradeoffs for your use case
- Profile with realistic data sets

### Common Pitfalls
- Don't use Fibonacci Heap for simple priority queue needs
- Avoid persistent structures for purely mutable algorithms
- Remember that amortized complexity may have worst-case spikes
- Consider the constant factors in big-O analysis

These advanced data structures provide the building blocks for sophisticated algorithms and can significantly improve performance when chosen appropriately for the problem domain.
