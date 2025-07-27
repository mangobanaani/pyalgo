"""
Test cases for Randomized algorithms
"""

import pytest
import random
from randomized.randomized_quicksort import randomized_quicksort, randomized_select
from randomized.skip_list import SkipList
from randomized.bloom_filter import BloomFilter
from randomized.reservoir_sampling import reservoir_sampling
from randomized.monte_carlo import monte_carlo_pi, monte_carlo_integration

class TestRandomizedQuicksort:
    def test_randomized_quicksort_basic(self):
        """Test basic randomized quicksort."""
        arr = [64, 34, 25, 12, 22, 11, 90]
        expected = sorted(arr)
        
        result = randomized_quicksort(arr.copy())
        assert result == expected
    
    def test_randomized_quicksort_empty(self):
        """Test with empty array."""
        assert randomized_quicksort([]) == []
    
    def test_randomized_quicksort_single(self):
        """Test with single element."""
        assert randomized_quicksort([42]) == [42]
    
    def test_randomized_quicksort_duplicates(self):
        """Test with duplicate elements."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        expected = sorted(arr)
        
        result = randomized_quicksort(arr.copy())
        assert result == expected
    
    def test_randomized_select(self):
        """Test randomized selection algorithm."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        
        # Find median (5th smallest in 0-indexed)
        result = randomized_select(arr.copy(), 4)
        sorted_arr = sorted(arr)
        assert result == sorted_arr[4]
        
        # Find minimum
        result = randomized_select(arr.copy(), 0)
        assert result == min(arr)
        
        # Find maximum
        result = randomized_select(arr.copy(), len(arr) - 1)
        assert result == max(arr)

class TestSkipList:
    def test_skip_list_basic_operations(self):
        """Test basic skip list operations."""
        skip_list = SkipList(max_level=4)
        
        # Test insertion
        skip_list.insert(3)
        skip_list.insert(1)
        skip_list.insert(4)
        skip_list.insert(2)
        
        # Test search
        assert skip_list.search(3)
        assert skip_list.search(1)
        assert not skip_list.search(5)
        
        # Test deletion
        skip_list.delete(2)
        assert not skip_list.search(2)
        assert skip_list.search(3)  # Others should still be there
    
    def test_skip_list_ordered_traversal(self):
        """Test that skip list maintains order."""
        skip_list = SkipList(max_level=4)
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        
        for val in values:
            skip_list.insert(val)
        
        # Get all values
        result = skip_list.get_all_values()
        assert result == sorted(set(values))  # Should be sorted and unique
    
    def test_skip_list_empty(self):
        """Test skip list when empty."""
        skip_list = SkipList(max_level=4)
        
        assert not skip_list.search(1)
        skip_list.delete(1)  # Should not crash
        assert skip_list.get_all_values() == []

class TestBloomFilter:
    def test_bloom_filter_basic(self):
        """Test basic bloom filter operations."""
        bf = BloomFilter(size=1000, hash_count=3)
        
        # Add elements
        bf.add("hello")
        bf.add("world")
        bf.add("python")
        
        # Test membership
        assert bf.contains("hello")
        assert bf.contains("world")
        assert bf.contains("python")
        
        # Test non-membership (might have false positives)
        # We can't guarantee this won't be a false positive
        # assert not bf.contains("nonexistent")
    
    def test_bloom_filter_false_positive_rate(self):
        """Test approximate false positive rate."""
        bf = BloomFilter(size=1000, hash_count=3)
        
        # Add some elements
        for i in range(100):
            bf.add(f"element_{i}")
        
        # Test false positives
        false_positives = 0
        test_count = 1000
        
        for i in range(100, 100 + test_count):
            if bf.contains(f"element_{i}"):
                false_positives += 1
        
        false_positive_rate = false_positives / test_count
        # Should be reasonably low (less than 10% for this configuration)
        assert false_positive_rate < 0.1

class TestReservoirSampling:
    def test_reservoir_sampling_basic(self):
        """Test basic reservoir sampling."""
        population = list(range(100))
        sample_size = 10
        
        sample = reservoir_sampling(population, sample_size)
        
        assert len(sample) == sample_size
        assert all(item in population for item in sample)
        assert len(set(sample)) == sample_size  # All unique
    
    def test_reservoir_sampling_small_population(self):
        """Test when population is smaller than sample size."""
        population = [1, 2, 3]
        sample_size = 5
        
        sample = reservoir_sampling(population, sample_size)
        
        assert len(sample) == len(population)
        assert set(sample) == set(population)
    
    def test_reservoir_sampling_distribution(self):
        """Test that reservoir sampling has uniform distribution."""
        population = list(range(100))
        sample_size = 10
        counts = [0] * 100
        iterations = 1000
        
        # Run many times and count occurrences
        for _ in range(iterations):
            sample = reservoir_sampling(population, sample_size)
            for item in sample:
                counts[item] += 1
        
        # Each item should appear roughly (sample_size * iterations / population_size) times
        expected_count = sample_size * iterations / len(population)
        
        # Check that distribution is reasonably uniform (within 50% of expected)
        for count in counts:
            assert abs(count - expected_count) < expected_count * 0.5

class TestMonteCarlo:
    def test_monte_carlo_pi(self):
        """Test Monte Carlo estimation of π."""
        random.seed(42)  # For reproducible results
        
        estimate = monte_carlo_pi(10000)
        
        # Should be reasonably close to π
        assert abs(estimate - 3.14159) < 0.1
    
    def test_monte_carlo_integration(self):
        """Test Monte Carlo integration."""
        random.seed(42)  # For reproducible results
        
        # Integrate x^2 from 0 to 1 (should be 1/3)
        def f(x):
            return x * x
        
        estimate = monte_carlo_integration(f, 0, 1, 10000)
        
        # Should be close to 1/3 ≈ 0.333
        assert abs(estimate - 1/3) < 0.05
    
    def test_monte_carlo_integration_constant(self):
        """Test Monte Carlo integration with constant function."""
        random.seed(42)
        
        # Integrate constant function 5 from 0 to 2 (should be 10)
        def f(x):
            return 5
        
        estimate = monte_carlo_integration(f, 0, 2, 10000)
        
        # Should be close to 10
        assert abs(estimate - 10) < 0.5

class TestRandomizedPerformance:
    def test_randomized_vs_deterministic_quicksort(self):
        """Compare randomized vs deterministic performance on worst-case input."""
        # Sorted array is worst case for deterministic quicksort
        arr = list(range(1000))
        
        # Both should produce correct result
        result_random = randomized_quicksort(arr.copy())
        assert result_random == arr
        
        # Test with reverse sorted
        arr_reverse = list(reversed(arr))
        result_random_reverse = randomized_quicksort(arr_reverse.copy())
        assert result_random_reverse == arr
