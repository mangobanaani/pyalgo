# Test Suite Documentation

This directory contains comprehensive test suites for all algorithms and data structures in the PyAlgo project. The tests ensure correctness, performance, and robustness of implementations.

## Test Structure

```
tests/
├── test_mergeSort.py          # Merge Sort algorithm tests
├── test_queue.py              # Queue data structure tests  
├── test_stack.py              # Stack data structure tests
├── test_advanced_data_structures.py  # Advanced data structures test suite
├── test_machine_learning.py   # Machine learning algorithms test suite
└── test_optimization.py       # Optimization algorithms test suite
```

## Test Categories

### Basic Data Structures Tests

**test_stack.py**
- Stack operations (push, pop, peek)
- Empty stack handling
- Stack overflow and underflow conditions
- LIFO behavior verification

**test_queue.py**
- Queue operations (enqueue, dequeue, front, rear)
- Circular queue implementation
- Empty queue handling
- FIFO behavior verification

**test_mergeSort.py**
- Sorting correctness on various input types
- Edge cases (empty arrays, single elements)
- Performance characteristics
- Stability verification

### Advanced Data Structures Tests

**test_advanced_data_structures.py** - Comprehensive test suite covering:

**Self-Balancing Trees**:
- AVL Tree: Insertion, deletion, rotation, balance factor verification
- Red-Black Tree: Color properties, tree balance, rotation operations
- Splay Tree: Splaying operations, access pattern optimization

**Heap Structures**:
- Binary Heap: Min/max heap properties, heap operations
- Fibonacci Heap: Advanced heap operations, decrease key, union
- Binomial Heap: Binomial tree properties, merge operations

**Specialized Structures**:
- Union-Find: Path compression, union by rank, connectivity queries
- Fenwick Tree: Range queries, point updates, prefix sums
- Persistent Array: Immutability, version control, structural sharing
- Persistent Segment Tree: Range queries across versions

**Integration Tests**:
- Cross-structure compatibility
- Performance benchmarks
- Memory usage analysis

### Machine Learning Tests

**test_machine_learning.py** - Comprehensive ML algorithm testing:

**Regression Algorithms**:
- Linear Regression: Simple and multiple regression, R² calculation
- Ridge Regression: Regularization effects, coefficient shrinkage
- Lasso Regression: Feature selection, sparsity induction

**Classification Algorithms**:
- Decision Tree Classifier: Splitting criteria, pruning, overfitting
- Neural Network Classifier: Backpropagation, activation functions
- SVM Classifier: Kernel functions, support vector identification
- Ensemble Methods: Random Forest, AdaBoost voting mechanisms

**Clustering Algorithms**:
- K-Means: Cluster assignment, convergence, initialization methods

**Model Evaluation**:
- Cross-validation techniques
- Performance metrics (accuracy, precision, recall, F1-score)
- Regression metrics (R², MSE, MAE)
- Clustering metrics (inertia, silhouette score)

**Edge Cases**:
- Empty datasets
- Single sample datasets
- High-dimensional data
- Perfectly separable data
- Linearly non-separable data

### Optimization Tests

**test_optimization.py** - Metaheuristic and optimization algorithm testing:

**Evolutionary Algorithms**:
- Genetic Algorithm: Selection, crossover, mutation operators
- Binary GA: Chromosome representation, genetic operators
- Real-valued GA: Continuous optimization, boundary handling

**Metaheuristics**:
- Simulated Annealing: Cooling schedules, acceptance probabilities
- Particle Swarm Optimization: Velocity updates, topology effects
- Differential Evolution: Mutation strategies, parameter adaptation

**Convergence Analysis**:
- Optimization progress tracking
- Convergence criteria testing
- Performance on standard test functions

**Parameter Sensitivity**:
- Algorithm parameter effects
- Adaptive parameter strategies
- Random seed reproducibility

## Test Functions and Benchmarks

### Standard Test Functions

**Optimization Test Functions**:
- Sphere Function: Simple unimodal optimization
- Rosenbrock Function: Classic optimization challenge
- Rastrigin Function: Highly multimodal test function
- Ackley Function: Multimodal with single global optimum
- Griewank Function: Multiple local optima

**Machine Learning Datasets**:
- Linear regression: y = mx + b relationships
- Polynomial regression: Non-linear relationships
- Classification: Linearly separable and non-separable data
- Clustering: Well-separated and overlapping clusters

### Performance Benchmarks

**Time Complexity Verification**:
- Sorting algorithms: O(n log n) vs O(n²) verification
- Search algorithms: O(log n) vs O(n) comparison
- Tree operations: O(log n) balance maintenance
- Graph algorithms: O(V + E) traversal complexity

**Space Complexity Analysis**:
- Memory usage patterns
- Garbage collection behavior
- Persistent structure overhead
- Algorithm space requirements

## Running Tests

### Individual Test Files

```bash
# Run specific test file
python -m pytest tests/test_machine_learning.py -v

# Run with coverage
python -m pytest tests/test_optimization.py --cov=optimization

# Run performance tests only
python -m pytest tests/test_advanced_data_structures.py -k "performance"
```

### Test Categories

```bash
# Run all data structure tests
python -m pytest tests/test_*data_structures*.py

# Run all algorithm tests
python -m pytest tests/test_*machine_learning*.py tests/test_*optimization*.py

# Run quick tests (skip performance benchmarks)
python -m pytest tests/ -k "not performance"
```

### Complete Test Suite

```bash
# Run all tests with detailed output
python -m pytest tests/ -v

# Run all tests with coverage report
python -m pytest tests/ --cov=. --cov-report=html

# Run tests in parallel
python -m pytest tests/ -n auto
```

## Test Design Principles

### Correctness Testing

1. **Algorithm Verification**: Each algorithm tested against known correct outputs
2. **Edge Case Coverage**: Empty inputs, single elements, boundary conditions
3. **Property Testing**: Mathematical properties and invariants verification
4. **Regression Testing**: Prevent introduction of bugs in existing functionality

### Performance Testing

1. **Complexity Verification**: Actual vs theoretical time/space complexity
2. **Scalability Testing**: Performance across different input sizes
3. **Memory Usage**: Memory leak detection and usage patterns
4. **Comparative Analysis**: Algorithm performance comparison

### Robustness Testing

1. **Input Validation**: Invalid input handling and error messages
2. **Numerical Stability**: Floating-point precision and overflow handling
3. **Resource Limits**: Memory and time limit behavior
4. **Random Testing**: Property-based testing with random inputs

## Test Utilities

### Custom Assertions

```python
# Algorithm-specific assertions
assert_tree_balanced(tree)          # Verify tree balance properties
assert_heap_property(heap)          # Verify heap ordering
assert_clustering_quality(clusters) # Verify cluster separation
assert_convergence(optimizer)       # Verify optimization convergence
```

### Test Data Generators

```python
# Generate test datasets
generate_random_array(size, min_val, max_val)
generate_linear_data(n_samples, noise_level)
generate_classification_data(n_classes, n_features)
generate_clustering_data(n_clusters, cluster_size)
```

### Performance Measuring

```python
# Time and memory measurement utilities
@measure_time
def test_sorting_performance():
    # Test implementation

@measure_memory
def test_memory_usage():
    # Memory usage test
```

## Continuous Integration

### Automated Testing

- **GitHub Actions**: Automatic test execution on commits
- **Multiple Python Versions**: Testing across Python 3.8+
- **Coverage Reports**: Minimum coverage requirements
- **Performance Regression**: Detect performance degradation

### Quality Gates

- **Code Coverage**: Minimum 85% line coverage
- **Test Success Rate**: 100% test pass rate required
- **Performance Benchmarks**: No significant performance regression
- **Documentation**: All public APIs documented and tested

## Test Data and Fixtures

### Standard Datasets

**Small Test Cases**:
- Hand-crafted minimal examples
- Known correct outputs
- Edge case scenarios

**Medium Test Cases**:
- Randomly generated data
- Realistic problem sizes
- Statistical validation

**Large Test Cases** (Performance only):
- Stress testing scenarios
- Scalability verification
- Resource limit testing

### Test Configuration

```python
# Test configuration constants
SMALL_SIZE = 100
MEDIUM_SIZE = 1000
LARGE_SIZE = 10000
PERFORMANCE_ITERATIONS = 5
RANDOM_SEED = 42
TOLERANCE = 1e-6
```

## Contributing Test Cases

### Adding New Tests

1. **Follow Naming Convention**: test_[algorithm]_[functionality].py
2. **Include Documentation**: Docstrings for test purpose and expected behavior
3. **Cover Edge Cases**: Empty inputs, single elements, boundary conditions
4. **Performance Tests**: Include timing and memory usage tests
5. **Random Testing**: Use fixed seeds for reproducibility

### Test Guidelines

1. **Isolated Tests**: Each test should be independent
2. **Clear Assertions**: Meaningful error messages
3. **Parameterized Tests**: Test multiple scenarios efficiently
4. **Setup and Teardown**: Proper resource management
5. **Documentation**: Comment complex test scenarios

This comprehensive test suite ensures the reliability, correctness, and performance of all PyAlgo implementations, providing confidence in the educational and practical use of these algorithms.
