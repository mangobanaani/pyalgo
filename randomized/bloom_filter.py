"""
Bloom Filter implementation.

A Bloom filter is a space-efficient probabilistic data structure that tests
whether an element is a member of a set. False positive matches are possible,
but false negatives are not.
"""

import hashlib
from typing import Union


class BloomFilter:
    """
    Bloom Filter data structure.
    
    A probabilistic data structure for testing set membership.
    Supports false positives but not false negatives.
    """
    
    def __init__(self, size: int, hash_count: int):
        """
        Initialize Bloom filter.
        
        Args:
            size: Size of the bit array
            hash_count: Number of hash functions to use
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size
        self.count = 0  # Number of elements added
    
    def _hash(self, item: Union[str, bytes], seed: int) -> int:
        """
        Hash function using SHA-256 with seed.
        
        Args:
            item: Item to hash
            seed: Seed for hash function
            
        Returns:
            Hash value modulo bit array size
        """
        if isinstance(item, str):
            item = item.encode('utf-8')
        
        # Combine item with seed
        hasher = hashlib.sha256()
        hasher.update(item)
        hasher.update(seed.to_bytes(4, byteorder='big'))
        
        # Return hash value modulo size
        return int(hasher.hexdigest(), 16) % self.size
    
    def add(self, item: Union[str, bytes]) -> None:
        """
        Add an item to the Bloom filter.
        
        Args:
            item: Item to add
            
        Time Complexity: O(k) where k is hash_count
        """
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
        
        self.count += 1
    
    def contains(self, item: Union[str, bytes]) -> bool:
        """
        Test if an item might be in the set.
        
        Args:
            item: Item to test
            
        Returns:
            True if item might be in set (possible false positive)
            False if item is definitely not in set
            
        Time Complexity: O(k) where k is hash_count
        """
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True
    
    def __contains__(self, item: Union[str, bytes]) -> bool:
        """Support 'in' operator."""
        return self.contains(item)
    
    def clear(self) -> None:
        """Clear all items from the Bloom filter."""
        self.bit_array = [False] * self.size
        self.count = 0
    
    def estimated_error_rate(self) -> float:
        """
        Estimate the current false positive rate.
        
        Returns:
            Estimated false positive probability
        """
        if self.count == 0:
            return 0.0
        
        # Calculate fraction of bits set to 1
        bits_set = sum(self.bit_array)
        fraction_set = bits_set / self.size
        
        # False positive rate = (fraction of bits set)^k
        return fraction_set ** self.hash_count
    
    def optimal_parameters(self, expected_items: int, 
                          false_positive_rate: float) -> tuple:
        """
        Calculate optimal parameters for given constraints.
        
        Args:
            expected_items: Expected number of items to be added
            false_positive_rate: Desired false positive rate
            
        Returns:
            Tuple of (optimal_size, optimal_hash_count)
        """
        import math
        
        # Optimal size: m = -(n * ln(p)) / (ln(2)^2)
        optimal_size = int(-(expected_items * math.log(false_positive_rate)) / 
                          (math.log(2) ** 2))
        
        # Optimal hash count: k = (m/n) * ln(2)
        optimal_hash_count = int((optimal_size / expected_items) * math.log(2))
        
        return optimal_size, max(1, optimal_hash_count)
    
    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Create union of two Bloom filters.
        
        Args:
            other: Another Bloom filter with same parameters
            
        Returns:
            New Bloom filter representing union
            
        Raises:
            ValueError: If filters have different parameters
        """
        if (self.size != other.size or 
            self.hash_count != other.hash_count):
            raise ValueError("Bloom filters must have same parameters for union")
        
        result = BloomFilter(self.size, self.hash_count)
        result.bit_array = [a or b for a, b in zip(self.bit_array, other.bit_array)]
        result.count = self.count + other.count  # Approximate
        
        return result
    
    def intersection(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Create intersection of two Bloom filters.
        
        Args:
            other: Another Bloom filter with same parameters
            
        Returns:
            New Bloom filter representing intersection
            
        Raises:
            ValueError: If filters have different parameters
        """
        if (self.size != other.size or 
            self.hash_count != other.hash_count):
            raise ValueError("Bloom filters must have same parameters for intersection")
        
        result = BloomFilter(self.size, self.hash_count)
        result.bit_array = [a and b for a, b in zip(self.bit_array, other.bit_array)]
        result.count = min(self.count, other.count)  # Approximate
        
        return result
    
    def __len__(self) -> int:
        """Return approximate number of items added."""
        return self.count
    
    def __str__(self) -> str:
        """String representation of Bloom filter."""
        return (f"BloomFilter(size={self.size}, hash_count={self.hash_count}, "
                f"items={self.count}, error_rate={self.estimated_error_rate():.4f})")


# Alternative implementation using built-in hash functions
class SimpleBloomFilter:
    """
    Simple Bloom filter using Python's built-in hash function.
    
    Less robust than the main BloomFilter but doesn't require bitarray.
    """
    
    def __init__(self, size: int, hash_count: int):
        """Initialize simple Bloom filter."""
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size
        self.count = 0
    
    def _hash(self, item: str, seed: int) -> int:
        """Simple hash function using built-in hash."""
        return hash(str(item) + str(seed)) % self.size
    
    def add(self, item: str) -> None:
        """Add item to filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
        self.count += 1
    
    def contains(self, item: str) -> bool:
        """Test if item might be in set."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True
