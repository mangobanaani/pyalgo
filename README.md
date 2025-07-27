# PyAlgo - Python Algorithms and Data Structures

A comprehensive collection of algorithms and data structures implemented in Python, organized by category and thoroughly tested.

## Project Structure

```
pyalgo/
├── sorting/                    # Sorting algorithms
│   ├── merge_sort.py          # Merge Sort (O(n log n))
│   ├── quick_sort.py          # Quick Sort (O(n log n) avg)
│   ├── bubble_sort.py         # Bubble Sort (O(n²))
│   └── heap_sort.py           # Heap Sort (O(n log n))
├── searching/                 # Search algorithms
│   ├── binary_search.py       # Binary Search (O(log n))
│   └── linear_search.py       # Linear Search (O(n))
├── data_structures/           # Data structure implementations
│   ├── Stack.py               # LIFO Stack
│   ├── Queue.py               # FIFO Queue (circular array)
│   ├── linked_list.py         # Singly Linked List
│   └── binary_search_tree.py  # Binary Search Tree
├── advanced_data_structures/  # Advanced data structures
│   ├── avl_tree.py           # Self-balancing AVL Tree
│   ├── red_black_tree.py     # Self-balancing Red-Black Tree
│   ├── splay_tree.py         # Self-adjusting Splay Tree
│   ├── binary_heap.py        # Binary Heap (priority queue)
│   ├── fibonacci_heap.py     # Fibonacci Heap
│   ├── binomial_heap.py      # Binomial Heap
│   ├── union_find.py         # Disjoint Set Union (Union-Find)
│   ├── fenwick_tree.py       # Binary Indexed Tree (Fenwick Tree)
│   ├── persistent_array.py   # Persistent Array
│   └── persistent_segment_tree.py # Persistent Segment Tree
├── string_algorithms/         # String processing algorithms
│   ├── kmp.py                 # Knuth-Morris-Pratt pattern matching
│   ├── boyer_moore.py         # Boyer-Moore pattern matching
│   ├── rabin_karp.py          # Rabin-Karp pattern matching
│   ├── z_algorithm.py         # Z-Algorithm for pattern matching
│   ├── aho_corasick.py        # Aho-Corasick multiple pattern matching
│   ├── trie.py                # Trie (prefix tree) implementation
│   ├── suffix_array.py        # Suffix Array implementation
│   ├── anagram_detection.py   # Algorithms for detecting anagrams
│   ├── palindrome_checking.py # Algorithms for checking palindromes
│   ├── string_compression.py  # Run-length encoding compression
│   ├── longest_common_substring.py # Find longest common substring
│   └── longest_palindromic_substring.py # Find longest palindromic substring
├── machine_learning/          # Machine learning algorithms
│   ├── linear_regression.py   # Linear, Ridge, and Lasso regression
│   ├── kmeans.py             # K-Means clustering
│   ├── neural_networks.py    # Neural networks and MLPs
│   ├── decision_trees.py     # Decision trees for classification/regression
│   ├── ensemble_methods.py   # Random Forest, AdaBoost
│   └── svm.py                # Support Vector Machines
├── optimization/              # Optimization algorithms
│   ├── genetic_algorithm.py   # Genetic algorithms (binary/real-valued)
│   ├── simulated_annealing.py # Simulated annealing variants
│   ├── particle_swarm.py     # Particle Swarm Optimization
│   └── differential_evolution.py # Differential Evolution
├── backtracking/              # Backtracking algorithms
│   ├── n_queens.py            # N-Queens problem
│   ├── sudoku_solver.py       # Sudoku solver
│   ├── knights_tour.py        # Knight's Tour problem
│   ├── maze_solver.py         # Maze solving problem
│   └── crossword_solver.py    # Crossword puzzle solver
├── greedy/                    # Greedy algorithms
│   ├── activity_selection.py  # Activity selection problem
│   ├── fractional_knapsack.py # Fractional Knapsack problem
│   └── huffman_coding.py      # Huffman coding for compression
├── dynamic_programming/       # Dynamic programming algorithms
│   └── dp_algorithms.py       # Fibonacci, LCS, Knapsack, etc.
├── graph/                     # Graph algorithms
│   └── graph_algorithms.py    # BFS, DFS, shortest path
├── tree_algorithms/           # Advanced tree algorithms
│   └── segment_tree.py        # Segment Tree implementation
├── mathematical/              # Mathematical & number theory algorithms
│   ├── number_theory.py       # GCD, primality, modular arithmetic
│   ├── linear_algebra.py      # Matrix operations, linear systems
│   ├── numerical_methods.py   # Root finding, integration, ODEs
│   ├── statistics.py          # Statistical analysis, regression
│   ├── combinatorics.py       # Factorials, permutations, sequences
│   ├── sieve_of_eratosthenes.py # Prime number generation
│   ├── gcd.py                 # Greatest Common Divisor
│   └── modular_arithmetic.py  # Fast modular operations
├── randomized/                # Randomized algorithms
│   ├── randomized_quicksort.py # Randomized Quick Sort
│   ├── skip_list.py           # Skip List data structure
│   ├── bloom_filter.py        # Bloom Filter for membership testing
│   ├── reservoir_sampling.py  # Reservoir sampling algorithm
│   └── monte_carlo.py         # Monte Carlo methods
├── cryptography/              # Cryptographic algorithms
│   ├── caesar_cipher.py       # Caesar cipher encryption/decryption
│   ├── vigenere_cipher.py     # Vigenère cipher implementation
│   ├── rsa.py                 # RSA public-key cryptography
│   └── aes.py                 # AES symmetric encryption
├── network_flow/              # Network flow algorithms
│   ├── max_flow.py            # Maximum flow algorithms
│   ├── min_cost_flow.py       # Minimum cost flow
│   └── matching.py            # Bipartite matching
├── geometry/                  # Computational geometry
│   ├── convex_hull.py         # Convex hull algorithms
│   ├── line_intersection.py   # Line segment intersection
│   └── closest_pair.py        # Closest pair of points
├── game_theory/               # Game theory algorithms
│   ├── minimax.py             # Minimax algorithm
│   ├── alpha_beta.py          # Alpha-beta pruning
│   └── game_tree.py           # Game tree structures
├── compression/               # Data compression algorithms
│   ├── huffman.py             # Huffman coding
│   ├── lz77.py                # LZ77 compression
│   └── rle.py                 # Run-length encoding
└── tests/                     # Comprehensive test suites
    ├── sorting/
    ├── searching/
    ├── data_structures/
    ├── string_algorithms/
    ├── backtracking/
    ├── greedy/
    ├── dynamic_programming/
    ├── graph/
    ├── tree_algorithms/
    ├── mathematical/
    ├── randomized/
    ├── cryptography/
    ├── network_flow/
    ├── geometry/
    ├── game_theory/
    └── compression/
```

## Algorithm Categories

### Sorting & Searching
- **Merge Sort, Quick Sort, Heap Sort, Bubble Sort** - Classic sorting algorithms
- **Binary Search, Linear Search** - Efficient searching techniques

### Data Structures  
- **Stack, Queue, Linked List** - Fundamental data structures
- **Binary Search Tree** - Hierarchical data organization

### Advanced Data Structures
- **Self-Balancing Trees** - AVL Tree, Red-Black Tree, Splay Tree
- **Advanced Heaps** - Binary Heap, Fibonacci Heap, Binomial Heap  
- **Union-Find** - Disjoint Set Union with path compression
- **Range Query Structures** - Fenwick Tree (Binary Indexed Tree)
- **Persistent Structures** - Persistent Array, Persistent Segment Tree

### Machine Learning
- **Supervised Learning** - Linear/Ridge/Lasso Regression, Decision Trees, SVM
- **Unsupervised Learning** - K-Means Clustering
- **Neural Networks** - Multi-layer Perceptrons, custom activation functions
- **Ensemble Methods** - Random Forest, AdaBoost

### Optimization Algorithms
- **Evolutionary Algorithms** - Genetic Algorithm (binary/real-valued)
- **Metaheuristics** - Simulated Annealing, Particle Swarm Optimization
- **Differential Evolution** - Multiple strategies, self-adaptive variants
- **Global Optimization** - Nature-inspired and population-based methods

### String Processing
- **Pattern Matching** - KMP, Boyer-Moore, Rabin-Karp, Z-Algorithm
- **Advanced Structures** - Trie, Suffix Array, Aho-Corasick
- **String Analysis** - Anagram detection, palindrome checking, compression

### Problem Solving Paradigms
- **Backtracking** - N-Queens, Sudoku, Knight's Tour, Maze solving
- **Greedy Algorithms** - Activity selection, Knapsack, Huffman coding
- **Dynamic Programming** - Fibonacci, LCS, optimization problems

### Graph & Trees
- **Graph Algorithms** - BFS, DFS, shortest paths
- **Advanced Trees** - Segment trees for range queries

### Mathematical Computing
- **Number Theory** - GCD, primality testing, modular arithmetic
- **Linear Algebra** - Matrix operations, system solving
- **Numerical Methods** - Root finding, integration, differential equations
- **Statistics** - Descriptive stats, regression, distributions
- **Combinatorics** - Factorials, permutations, special sequences

### Randomized Algorithms
- **Probabilistic Data Structures** - Skip Lists, Bloom Filters
- **Randomized Sorting** - Randomized QuickSort
- **Sampling** - Reservoir sampling for large datasets
- **Monte Carlo Methods** - Probabilistic computation

### Cryptography
- **Classical Ciphers** - Caesar, Vigenère encryption
- **Modern Cryptography** - RSA public-key, AES symmetric encryption

### Network Flow & Geometry
- **Flow Networks** - Maximum flow, minimum cost flow, bipartite matching
- **Computational Geometry** - Convex hull, line intersection, closest pairs

### Game Theory & AI
- **Game Trees** - Minimax algorithm with alpha-beta pruning
- **Decision Making** - Optimal strategies in competitive scenarios

### Data Compression
- **Lossless Compression** - Huffman coding, LZ77, Run-length encoding

## Sorting Algorithms

### Merge Sort
- **Time Complexity**: O(n log n) in all cases
- **Space Complexity**: O(n)
- **Stable**: Yes
- **Use Case**: When you need guaranteed O(n log n) performance

### Quick Sort
- **Time Complexity**: O(n log n) average, O(n²) worst case
- **Space Complexity**: O(log n) average
- **Stable**: No
- **Use Case**: General purpose sorting, often fastest in practice

### Bubble Sort
- **Time Complexity**: O(n²)
- **Space Complexity**: O(1)
- **Stable**: Yes
- **Use Case**: Educational purposes, small datasets

### Heap Sort
- **Time Complexity**: O(n log n) in all cases
- **Space Complexity**: O(1)
- **Stable**: No
- **Use Case**: When memory is limited

## Searching Algorithms

### Binary Search
- **Time Complexity**: O(log n)
- **Space Complexity**: O(1) iterative, O(log n) recursive
- **Requirement**: Sorted array
- **Variants**: Standard, leftmost, rightmost occurrence

### Linear Search
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)
- **Requirement**: None (works on unsorted arrays)
- **Features**: Find all occurrences, custom conditions

## Data Structures

### Stack (LIFO)
- **Operations**: push, pop, top, is_empty
- **Time Complexity**: O(1) for all operations
- **Use Cases**: Function calls, expression evaluation, undo operations

### Queue (FIFO)
- **Operations**: enqueue, dequeue, first, is_empty
- **Implementation**: Circular array with dynamic resizing
- **Time Complexity**: O(1) amortized for all operations

### Linked List
- **Operations**: append, prepend, delete, find, reverse
- **Time Complexity**: O(1) for insertion, O(n) for search/deletion
- **Space Complexity**: O(n)

### Binary Search Tree
- **Operations**: insert, search, delete, traversals
- **Time Complexity**: O(log n) average, O(n) worst case
- **Traversals**: Inorder, preorder, postorder

## String Algorithms

### Pattern Matching
- **KMP (Knuth-Morris-Pratt)**: O(n+m) pattern searching
- **Boyer-Moore**: Efficient pattern matching with bad character heuristic
- **Rabin-Karp**: Rolling hash pattern matching
- **Z-Algorithm**: Linear time pattern matching
- **Aho-Corasick**: Multiple pattern matching

### String Processing
- **Longest Common Substring**: Find longest common substring
- **Longest Palindromic Substring**: Manacher's algorithm
- **String Compression**: Run-length encoding
- **Anagram Detection**: Various approaches to detect anagrams
- **Palindrome Checking**: Efficient palindrome verification

### Advanced String Structures
- **Trie (Prefix Tree)**: Efficient string storage and retrieval
- **Suffix Array**: Space-efficient suffix sorting

## Backtracking Algorithms

### Classic Puzzles
- **N-Queens Problem**: Place N queens on chessboard
- **Sudoku Solver**: Complete Sudoku puzzles
- **Knight's Tour**: Visit all squares on chessboard
- **Maze Solving**: Find path through maze
- **Crossword Puzzle**: Fill crossword grids

## Greedy Algorithms

### Classic Problems
- **Activity Selection**: Maximum non-overlapping activities
- **Fractional Knapsack**: Continuous knapsack problem
- **Huffman Coding**: Optimal prefix codes

## Mathematical Algorithms

### Number Theory & Arithmetic
- **Prime Numbers**: Sieve of Eratosthenes, primality testing
- **GCD & LCM**: Euclidean algorithm, extended GCD
- **Modular Arithmetic**: Fast exponentiation, modular inverse, Chinese Remainder Theorem
- **Number Sequences**: Fibonacci, factorials, Euler's totient

### Linear Algebra & Systems
- **Matrix Operations**: Addition, multiplication, determinant, inverse
- **Linear Systems**: Gaussian elimination, LU decomposition, QR decomposition
- **Vector Operations**: Dot product, cross product, norms

### Numerical Methods & Analysis
- **Root Finding**: Newton-Raphson, bisection method
- **Integration**: Trapezoidal rule, Simpson's rule
- **Differential Equations**: Euler's method, Runge-Kutta methods
- **Optimization**: Gradient descent, numerical optimization

### Statistics & Probability
- **Descriptive Statistics**: Mean, median, mode, variance, standard deviation
- **Correlation & Regression**: Pearson correlation, linear regression
- **Probability Distributions**: Normal, binomial, chi-square distributions
- **Hypothesis Testing**: Statistical significance testing

### Combinatorics & Sequences
- **Counting**: Factorials, permutations, combinations
- **Special Numbers**: Catalan numbers, Stirling numbers, Bell numbers
- **Integer Sequences**: Fibonacci, partition functions
- **Advanced Combinatorics**: Generating functions, recurrence relations

## Randomized Algorithms

### Probabilistic Data Structures
- **Skip List**: Probabilistic balanced tree alternative
- **Bloom Filter**: Space-efficient probabilistic membership testing

### Randomized Sorting & Selection
- **Randomized QuickSort**: Average-case O(n log n) sorting with random pivot
- **Random Sampling**: Efficient sampling from large datasets

### Monte Carlo & Probabilistic Methods
- **Monte Carlo Integration**: Numerical integration using random sampling
- **Reservoir Sampling**: Uniform sampling from streams of unknown size
- **Probabilistic Algorithms**: Las Vegas vs Monte Carlo approaches

## Cryptography

### Classical Cryptography
- **Caesar Cipher**: Simple substitution cipher with shift
- **Vigenère Cipher**: Polyalphabetic substitution cipher
- **Frequency Analysis**: Cryptanalysis techniques

### Modern Cryptography
- **RSA**: Public-key cryptography with key generation
- **AES**: Advanced Encryption Standard (symmetric)
- **Key Management**: Secure key generation and exchange

## Network Flow Algorithms

### Maximum Flow
- **Ford-Fulkerson**: Basic maximum flow algorithm
- **Edmonds-Karp**: BFS-based maximum flow
- **Dinic's Algorithm**: Efficient maximum flow with blocking flows

### Specialized Flow Problems
- **Minimum Cost Flow**: Cost optimization in flow networks
- **Bipartite Matching**: Maximum matching in bipartite graphs
- **Multi-commodity Flow**: Multiple flow types in single network

## Computational Geometry

### Fundamental Problems
- **Convex Hull**: Graham scan and other hull algorithms
- **Line Intersection**: Detecting and computing intersections
- **Closest Pair**: Efficient closest pair of points

### Advanced Geometric Algorithms
- **Point Location**: Determining point position relative to polygons
- **Triangulation**: Decomposing polygons into triangles
- **Voronoi Diagrams**: Proximity-based space partitioning

## Game Theory & AI

### Game Tree Algorithms
- **Minimax**: Optimal play in zero-sum games
- **Alpha-Beta Pruning**: Efficient minimax with pruning
- **Game Tree Evaluation**: Position evaluation and heuristics

### Strategic Decision Making
- **Nash Equilibrium**: Finding stable strategy profiles
- **Mechanism Design**: Designing games with desired outcomes
- **Auction Theory**: Optimal bidding strategies

## Data Compression

### Lossless Compression
- **Huffman Coding**: Optimal prefix codes for data compression
- **LZ77**: Dictionary-based compression algorithm
- **Run-Length Encoding**: Simple repetition-based compression

### Compression Analysis
- **Entropy Calculation**: Information-theoretic compression limits
- **Compression Ratio**: Measuring compression effectiveness

## Advanced Tree Algorithms

### Specialized Trees
- **Segment Tree**: Range query and update operations

## Graph Algorithms

### Graph Representation
- **Adjacency List**: Space-efficient for sparse graphs
- **Support**: Directed and undirected graphs, weighted edges

### Algorithms
- **BFS (Breadth-First Search)**: Level-order traversal, shortest path in unweighted graphs
- **DFS (Depth-First Search)**: Recursive and iterative implementations
- **Path Finding**: Check connectivity, find shortest paths

## Dynamic Programming

### Classic Problems
- **Fibonacci**: Memoization and tabulation approaches
- **Longest Common Subsequence (LCS)**: String comparison
- **0/1 Knapsack**: Optimization problem
- **Coin Change**: Minimum coins for amount
- **Longest Increasing Subsequence**: Array analysis
- **Edit Distance**: String similarity (Levenshtein distance)

## Getting Started

### Installation

#### Using Poetry (Recommended)
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/mangobanaani/pyalgo.git
cd pyalgo

# Install dependencies
poetry install

# Run demos and benchmarks
poetry run python demo.py
poetry run python benchmark.py

# Run tests
poetry run pytest tests/ -v
```

#### Using pip
```bash
# Clone the repository
git clone https://github.com/mangobanaani/pyalgo.git
cd pyalgo

# Install requirements
pip install -r requirements.txt

# Run demos and benchmarks
python demo.py
python benchmark.py
```

### Development Setup
```bash
# Install with development dependencies
poetry install --with=dev,benchmark

# Run code formatting
poetry run black .
poetry run isort .

# Run linting
poetry run flake8 .
poetry run mypy . --ignore-missing-imports

# Run tests with coverage
poetry run pytest tests/ --cov=. --cov-report=html
```

### Using the Makefile
```bash
make help          # Show available commands
make install-dev   # Install all dependencies
make test          # Run tests
make format        # Format code
make benchmark     # Run performance benchmarks
make demo          # Run algorithm demonstrations
```

## Usage Examples

### Sorting
```python
from sorting.quick_sort import QuickSort
from sorting.merge_sort import MergeSort

arr = [3, 1, 4, 1, 5, 9, 2, 6]
QuickSort.quick_sort(arr)
print(arr)  # [1, 1, 2, 3, 4, 5, 6, 9]
```

### Mathematical Algorithms
```python
from mathematical.number_theory import gcd, is_prime
from mathematical.linear_algebra import Matrix
from mathematical.statistics import mean, correlation

# Number theory
print(gcd(48, 18))  # 6
print(is_prime(17))  # True

# Linear algebra
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[2, 0], [1, 2]])
C = A + B  # Matrix addition

# Statistics
data = [1, 2, 3, 4, 5]
print(mean(data))  # 3.0
```

### Randomized Algorithms
```python
from randomized.skip_list import SkipList
from randomized.bloom_filter import BloomFilter
from randomized.reservoir_sampling import reservoir_sample

# Skip List
skip_list = SkipList()
skip_list.insert(5)
skip_list.insert(3)
print(skip_list.search(5))  # True

# Bloom Filter
bloom = BloomFilter(1000, 0.01)
bloom.add("hello")
print(bloom.contains("hello"))  # True (definitely)
print(bloom.contains("world"))  # False (probably)
```

### Cryptography
```python
from cryptography.caesar_cipher import caesar_encrypt, caesar_decrypt, caesar_crack
from cryptography.rsa import generate_keypair, rsa_encrypt, rsa_decrypt

# Caesar Cipher
encrypted = caesar_encrypt("HELLO", 3)
print(encrypted)  # "KHOOR"
decrypted = caesar_decrypt(encrypted, 3)
print(decrypted)  # "HELLO"

# Crack cipher
plaintext, shift = caesar_crack("KHOOR")
print(f"Cracked: {plaintext} with shift {shift}")
```

### Data Structures
```python
from data_structures.Stack import Stack
from data_structures.binary_search_tree import BinarySearchTree

# Stack usage
stack = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())  # 2

# BST usage
bst = BinarySearchTree()
bst.insert(5)
bst.insert(3)
bst.insert(7)
print(bst.inorder_traversal())  # [3, 5, 7]
```

### Searching
```python
from searching.binary_search import BinarySearch
from searching.linear_search import LinearSearch

sorted_arr = [1, 3, 5, 7, 9, 11]
index = BinarySearch.binary_search(sorted_arr, 7)
print(index)  # 3
```

### Graph Algorithms
```python
from graph.graph_algorithms import Graph, GraphAlgorithms

graph = Graph()
graph.add_edge('A', 'B')
graph.add_edge('B', 'C')
graph.add_edge('A', 'C')

traversal = GraphAlgorithms.bfs(graph, 'A')
print(traversal)  # ['A', 'B', 'C']
```

### Dynamic Programming
```python
from dynamic_programming.dp_algorithms import DynamicProgramming

# Fibonacci
fib_10 = DynamicProgramming.fibonacci_memo(10)
print(fib_10)  # 55

# Longest Common Subsequence
lcs_length = DynamicProgramming.longest_common_subsequence("ABCDGH", "AEDFHR")
print(lcs_length)  # 3
```

## Time Complexity Summary

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| BST Search | O(log n) | O(log n) | O(n) | O(n) |

## Performance Benchmarks

### Sorting Algorithms (Random Data)

| Array Size | Merge Sort | Quick Sort | Heap Sort | Bubble Sort |
|------------|------------|------------|-----------|-------------|
| 100 | 0.000157s | 0.000054s | 0.000130s | 0.000304s |
| 500 | 0.000777s | 0.000357s | 0.000899s | 0.008350s |
| 1,000 | 0.001597s | 0.000769s | 0.002080s | 0.034175s |
| 5,000 | 0.010056s | 0.004901s | 0.013002s | Too slow |

**Key Insights:**
- **Quick Sort** consistently shows the best performance for random data
- **Bubble Sort** becomes impractical for arrays larger than 1,000 elements
- **Merge Sort** provides stable O(n log n) performance

### Searching Algorithms

| Array Size | Binary Search | Linear Search | Speedup Factor |
|------------|---------------|---------------|----------------|
| 1,000 | 0.00000067s | 0.00001303s | 19.3x |
| 10,000 | 0.00000038s | 0.00012421s | 323.9x |
| 100,000 | 0.00000047s | 0.00125796s | 2,694.4x |
| 1,000,000 | 0.00000248s | 0.01243902s | 5,009.1x |

**Key Insights:**
- Binary search speedup increases dramatically with data size
- For 1M elements, binary search is **5,000x faster** than linear search

*Run `python benchmark.py` to generate fresh performance data on your system.*

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings and comments
3. Include test cases for new algorithms
4. Update the README with new additions

## License

This project is open source and available under the MIT License.
