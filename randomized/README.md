# Randomized Algorithms

Collection of probabilistic algorithms and data structures that use randomization to achieve efficient performance or provide approximate solutions.

## Algorithms Included

### Randomized QuickSort (`randomized_quicksort.py`)
- **Description**: QuickSort variant that randomly selects pivot to avoid worst-case O(n²) performance
- **Time Complexity**: O(n log n) expected, O(n²) worst case (very unlikely)
- **Space Complexity**: O(log n) expected
- **Advantage**: Eliminates dependency on input order, provides consistent expected performance
- **Use Case**: General-purpose sorting when consistent performance is critical

### Skip List (`skip_list.py`)
- **Description**: Probabilistic data structure that maintains sorted elements with fast search, insertion, and deletion
- **Time Complexity**: O(log n) expected for all operations
- **Space Complexity**: O(n) expected
- **Advantage**: Simpler implementation than balanced trees, good cache performance
- **Use Case**: Alternative to balanced binary search trees, database indexing

### Bloom Filter (`bloom_filter.py`)
- **Description**: Space-efficient probabilistic data structure for membership testing
- **Time Complexity**: O(k) where k is number of hash functions
- **Space Complexity**: O(m) where m is bit array size
- **Properties**: No false negatives, possible false positives with tunable probability
- **Use Case**: Web crawling, database query optimization, distributed systems

### Reservoir Sampling (`reservoir_sampling.py`)
- **Description**: Algorithm for sampling k elements from stream of n elements where n is unknown
- **Time Complexity**: O(n) single pass through data
- **Space Complexity**: O(k) for storing k samples
- **Advantage**: Works with streams of unknown/infinite size
- **Use Case**: Sampling from large datasets, streaming analytics

### Monte Carlo Methods (`monte_carlo.py`)
- **Description**: Computational algorithms using repeated random sampling
- **Includes**: Pi estimation, numerical integration, random walk simulation
- **Convergence**: Error decreases as O(1/√n) where n is number of samples
- **Use Case**: Numerical integration, optimization, simulation

## Usage Examples

### Randomized QuickSort
```python
from randomized.randomized_quicksort import randomized_quicksort

# Sort array with random pivot selection
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
randomized_quicksort(arr)
print(f"Sorted: {arr}")

# Compare with worst-case input for regular quicksort
worst_case = list(range(1000, 0, -1))  # Reverse sorted
randomized_quicksort(worst_case)  # Still O(n log n) expected
```

### Skip List
```python
from randomized.skip_list import SkipList

# Create skip list and perform operations
skip_list = SkipList()

# Insert elements
skip_list.insert(3)
skip_list.insert(1)
skip_list.insert(4)
skip_list.insert(2)

# Search elements
print(skip_list.search(3))  # True
print(skip_list.search(5))  # False

# Delete element
skip_list.delete(1)
print(skip_list.search(1))  # False

# Display structure
skip_list.display()
```

### Bloom Filter
```python
from randomized.bloom_filter import BloomFilter

# Create Bloom filter with 1000 capacity and 1% false positive rate
bloom = BloomFilter(capacity=1000, false_positive_rate=0.01)

# Add elements
bloom.add("apple")
bloom.add("banana")
bloom.add("orange")

# Test membership
print(bloom.contains("apple"))   # True (definitely present)
print(bloom.contains("grape"))   # False (definitely not present)
print(bloom.contains("cherry"))  # False (probably not present)

# Check filter statistics
print(f"Bit array size: {bloom.bit_array_size}")
print(f"Number of hash functions: {bloom.num_hash_functions}")
print(f"Expected false positive rate: {bloom.false_positive_rate}")
```

### Reservoir Sampling
```python
from randomized.reservoir_sampling import reservoir_sample

# Sample from a large stream
def data_stream():
    """Simulate streaming data"""
    for i in range(10000):
        yield f"item_{i}"

# Sample 10 items from stream
sample = reservoir_sample(data_stream(), k=10)
print(f"Random sample: {sample}")

# Sample from list
large_list = list(range(1000))
sample = reservoir_sample(iter(large_list), k=5)
print(f"Sample from list: {sample}")
```

### Monte Carlo Methods
```python
from randomized.monte_carlo import estimate_pi, monte_carlo_integration

# Estimate π using random points
pi_estimate = estimate_pi(100000)
print(f"π estimate: {pi_estimate}")

# Numerical integration using Monte Carlo
def f(x):
    return x * x  # Integrate x² from 0 to 1 (should be ≈ 1/3)

integral = monte_carlo_integration(f, 0, 1, 100000)
print(f"Integral estimate: {integral}")

# Random walk simulation
from randomized.monte_carlo import random_walk_2d
final_position = random_walk_2d(1000)
print(f"Final position after 1000 steps: {final_position}")
```

## Algorithm Analysis

### Performance Characteristics

| Algorithm | Time Complexity | Space Complexity | Success Probability |
|-----------|----------------|------------------|-------------------|
| Randomized QuickSort | O(n log n) expected | O(log n) expected | High |
| Skip List | O(log n) expected | O(n) expected | High |
| Bloom Filter | O(k) per operation | O(m) total | Perfect (no false negatives) |
| Reservoir Sampling | O(n) single pass | O(k) | Perfect (uniform sampling) |
| Monte Carlo | O(n) samples | O(1) | Improves with more samples |

### Probabilistic Guarantees

- **Randomized QuickSort**: Worst-case O(n²) probability is O(1/n!)
- **Skip List**: Operation takes > c log n time with probability O(1/n^c)
- **Bloom Filter**: False positive rate is configurable and predictable
- **Reservoir Sampling**: Each element has exactly k/n probability of selection
- **Monte Carlo**: Standard error decreases as 1/√n

## Applications

### Randomized QuickSort
- General-purpose sorting in libraries (Java's Arrays.sort)
- Database query processing
- External sorting algorithms

### Skip List
- Redis sorted sets implementation
- LevelDB and other LSM-tree databases
- Concurrent data structures

### Bloom Filter
- Web crawlers (avoiding revisiting URLs)
- Database query optimization
- Distributed caching systems
- Network routing protocols

### Reservoir Sampling
- A/B testing with streaming data
- Log analysis and monitoring
- Social media feed sampling
- Survey sampling from large populations

### Monte Carlo Methods
- Financial risk modeling
- Physics simulations
- Machine learning (dropout, MCMC)
- Game AI and decision making

## When to Use Randomized Algorithms

**Choose randomized algorithms when:**
- Average-case performance is more important than worst-case
- Simple implementation is preferred over complex deterministic solutions
- Dealing with adversarial inputs or unknown data distributions
- Approximate solutions are acceptable
- Memory efficiency is crucial (Bloom filters)
- Processing streaming data with unknown size

**Avoid when:**
- Deterministic behavior is required
- Real-time systems with strict timing constraints
- Cryptographic applications (unless using cryptographically secure randomness)
- Small datasets where overhead isn't justified
