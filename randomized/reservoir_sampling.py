"""
Reservoir Sampling algorithm.

Reservoir sampling is a family of randomized algorithms for choosing k samples
from a list of n items, where n is either a very large or unknown number.
"""

import random
from typing import List, TypeVar, Iterator

T = TypeVar('T')


def reservoir_sampling(population: List[T], k: int) -> List[T]:
    """
    Sample k items from population using reservoir sampling.
    
    Each item in the population has an equal probability of being selected
    regardless of the size of the population.
    
    Args:
        population: List of items to sample from
        k: Number of items to sample
        
    Returns:
        List of k sampled items (or all items if k > len(population))
        
    Time Complexity: O(n) where n is len(population)
    Space Complexity: O(k)
    """
    if k >= len(population):
        return population.copy()
    
    if k <= 0:
        return []
    
    # Initialize reservoir with first k elements
    reservoir = population[:k]
    
    # Process remaining elements
    for i in range(k, len(population)):
        # Generate random index from 0 to i (inclusive)
        j = random.randint(0, i)
        
        # If j is in range [0, k-1], replace reservoir[j] with population[i]
        if j < k:
            reservoir[j] = population[i]
    
    return reservoir


def streaming_reservoir_sampling(stream: Iterator[T], k: int) -> List[T]:
    """
    Sample k items from a stream using reservoir sampling.
    
    This version works with iterators/streams where the total size
    is not known in advance.
    
    Args:
        stream: Iterator of items to sample from
        k: Number of items to sample
        
    Returns:
        List of k sampled items
        
    Time Complexity: O(n) where n is the number of items in stream
    Space Complexity: O(k)
    """
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            # Fill reservoir with first k items
            reservoir.append(item)
        else:
            # Randomly replace items in reservoir
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return reservoir


def weighted_reservoir_sampling(items: List[tuple], k: int) -> List[T]:
    """
    Weighted reservoir sampling where each item has an associated weight.
    
    Items with higher weights are more likely to be selected.
    
    Args:
        items: List of (item, weight) tuples
        k: Number of items to sample
        
    Returns:
        List of k sampled items
        
    Time Complexity: O(n log k) where n is len(items)
    Space Complexity: O(k)
    """
    import heapq
    
    if k >= len(items):
        return [item for item, weight in items]
    
    if k <= 0:
        return []
    
    # Use a min-heap to maintain k items with largest keys
    heap = []
    
    for item, weight in items:
        if weight <= 0:
            continue
            
        # Generate random key: uniform(0,1)^(1/weight)
        key = random.random() ** (1.0 / weight)
        
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))
    
    return [item for key, item in heap]


def reservoir_sampling_with_replacement(population: List[T], k: int) -> List[T]:
    """
    Sample k items from population with replacement.
    
    Each draw is independent, so the same item can be selected multiple times.
    
    Args:
        population: List of items to sample from
        k: Number of items to sample
        
    Returns:
        List of k sampled items (may contain duplicates)
        
    Time Complexity: O(k)
    Space Complexity: O(k)
    """
    if not population:
        return []
    
    return [random.choice(population) for _ in range(k)]


def online_reservoir_sampling(k: int):
    """
    Create an online reservoir sampler that can be updated incrementally.
    
    Returns a sampler object with add() and get_sample() methods.
    
    Args:
        k: Size of the reservoir
        
    Returns:
        OnlineReservoirSampler instance
        
    Example:
        sampler = online_reservoir_sampling(3)
        for item in data_stream:
            sampler.add(item)
        sample = sampler.get_sample()
    """
    return OnlineReservoirSampler(k)


class OnlineReservoirSampler:
    """
    Online reservoir sampler for streaming data.
    
    Maintains a reservoir of k items that represents a uniform random
    sample of all items seen so far.
    """
    
    def __init__(self, k: int):
        """
        Initialize online reservoir sampler.
        
        Args:
            k: Size of the reservoir
        """
        self.k = k
        self.reservoir = []
        self.count = 0
    
    def add(self, item: T) -> None:
        """
        Add a new item to the stream.
        
        Args:
            item: Item to add
        """
        self.count += 1
        
        if len(self.reservoir) < self.k:
            # Reservoir not full, just add the item
            self.reservoir.append(item)
        else:
            # Reservoir is full, decide whether to replace an item
            j = random.randint(1, self.count)
            if j <= self.k:
                # Replace item at position j-1 (convert to 0-based index)
                self.reservoir[j - 1] = item
    
    def get_sample(self) -> List[T]:
        """
        Get the current sample.
        
        Returns:
            Copy of the current reservoir
        """
        return self.reservoir.copy()
    
    def size(self) -> int:
        """Get current reservoir size."""
        return len(self.reservoir)
    
    def total_count(self) -> int:
        """Get total number of items processed."""
        return self.count


def stratified_reservoir_sampling(items: List[tuple], k: int) -> List[T]:
    """
    Stratified reservoir sampling to ensure representation from different strata.
    
    Args:
        items: List of (item, stratum) tuples
        k: Total number of items to sample
        
    Returns:
        List of sampled items with proportional representation
        
    Note:
        This is a simplified version that ensures at least one item
        from each stratum if possible.
    """
    from collections import defaultdict
    
    # Group items by stratum
    strata = defaultdict(list)
    for item, stratum in items:
        strata[stratum].append(item)
    
    if not strata:
        return []
    
    # Allocate samples proportionally to strata sizes
    total_items = len(items)
    num_strata = len(strata)
    
    # Ensure at least one sample per stratum if k >= num_strata
    if k >= num_strata:
        samples_per_stratum = {
            stratum: max(1, int(k * len(stratum_items) / total_items))
            for stratum, stratum_items in strata.items()
        }
        
        # Adjust if we allocated too many
        total_allocated = sum(samples_per_stratum.values())
        if total_allocated > k:
            # Reduce from largest strata
            sorted_strata = sorted(strata.items(), key=lambda x: len(x[1]), reverse=True)
            for stratum, _ in sorted_strata:
                if total_allocated <= k:
                    break
                if samples_per_stratum[stratum] > 1:
                    samples_per_stratum[stratum] -= 1
                    total_allocated -= 1
    else:
        # If k < num_strata, sample randomly from strata
        selected_strata = random.sample(list(strata.keys()), k)
        samples_per_stratum = {stratum: 1 for stratum in selected_strata}
    
    # Sample from each stratum
    result = []
    for stratum, num_samples in samples_per_stratum.items():
        stratum_items = strata[stratum]
        stratum_sample = reservoir_sampling(stratum_items, num_samples)
        result.extend(stratum_sample)
    
    return result
