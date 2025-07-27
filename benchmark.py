#!/usr/bin/env python3
"""
Performance Benchmarking Suite for PyAlgo
Compares the performance of different algorithms and generates reports.
"""

import time
import random
import statistics
import sys
import os
from typing import List, Dict, Callable, Tuple

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from sorting import MergeSort, QuickSort, BubbleSort, HeapSort
from searching import BinarySearch, LinearSearch
from data_structures import Stack, Queue, LinkedList, BinarySearchTree


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> float:
        """Measure execution time of a function in seconds"""
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time
    
    @staticmethod
    def benchmark_function(func: Callable, data, num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark a function with multiple runs and return statistics
        
        Args:
            func: Function to benchmark
            data: Data to pass to the function
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        times = []
        for _ in range(num_runs):
            # Create a copy of data for each run to avoid side effects
            if hasattr(data, 'copy'):
                data_copy = data.copy()
            elif isinstance(data, (list, tuple)):
                if isinstance(data, tuple) and len(data) == 2:
                    # Handle (array, target) tuple for search functions
                    arr, target = data
                    data_copy = (arr.copy() if hasattr(arr, 'copy') else arr, target)
                else:
                    data_copy = data.copy() if hasattr(data, 'copy') else data
            else:
                data_copy = data
                
            execution_time = PerformanceBenchmark.measure_time(func, data_copy)
            times.append(execution_time)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'runs': num_runs
        }


class SortingBenchmark:
    """Benchmarking suite for sorting algorithms"""
    
    ALGORITHMS = {
        'Merge Sort': MergeSort.merge_sort,
        'Quick Sort': QuickSort.quick_sort,
        'Heap Sort': HeapSort.heap_sort,
        'Bubble Sort': BubbleSort.bubble_sort,
    }
    
    @staticmethod
    def generate_test_data(size: int, data_type: str = 'random') -> List[int]:
        """Generate test data for sorting algorithms"""
        if data_type == 'random':
            return [random.randint(1, size * 10) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(1, size + 1))
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'nearly_sorted':
            data = list(range(1, size + 1))
            # Shuffle 10% of elements
            num_swaps = max(1, size // 10)
            for _ in range(num_swaps):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                data[i], data[j] = data[j], data[i]
            return data
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    @staticmethod
    def benchmark_sorting_algorithms(sizes: List[int], data_types: List[str] = None) -> Dict:
        """Benchmark all sorting algorithms with different input sizes and types"""
        if data_types is None:
            data_types = ['random', 'sorted', 'reverse', 'nearly_sorted']
        
        results = {}
        
        for size in sizes:
            print(f"Benchmarking with array size: {size}")
            results[size] = {}
            
            for data_type in data_types:
                print(f"  Testing {data_type} data...")
                results[size][data_type] = {}
                
                for algo_name, algo_func in SortingBenchmark.ALGORITHMS.items():
                    # Skip bubble sort for large arrays (too slow)
                    if algo_name == 'Bubble Sort' and size > 1000:
                        results[size][data_type][algo_name] = {'mean': float('inf'), 'note': 'Skipped (too slow)'}
                        continue
                    
                    try:
                        test_data = SortingBenchmark.generate_test_data(size, data_type)
                        benchmark_result = PerformanceBenchmark.benchmark_function(
                            algo_func, test_data, num_runs=3
                        )
                        results[size][data_type][algo_name] = benchmark_result
                        print(f"    {algo_name}: {benchmark_result['mean']:.6f}s")
                    except Exception as e:
                        results[size][data_type][algo_name] = {'error': str(e)}
        
        return results


class SearchingBenchmark:
    """Benchmarking suite for searching algorithms"""
    
    @staticmethod
    def benchmark_search_algorithms(sizes: List[int]) -> Dict:
        """Benchmark searching algorithms"""
        results = {}
        
        for size in sizes:
            print(f"Benchmarking search with array size: {size}")
            results[size] = {}
            
            # Generate sorted array for binary search
            sorted_array = list(range(1, size + 1))
            # Target to search for (middle element)
            target = size // 2
            
            # Binary Search - create a lambda that captures both array and target
            def binary_search_test(arr_and_target):
                arr, tgt = arr_and_target
                return BinarySearch.binary_search(arr, tgt)
            
            binary_result = PerformanceBenchmark.benchmark_function(
                binary_search_test, (sorted_array, target)
            )
            results[size]['Binary Search'] = binary_result
            print(f"  Binary Search: {binary_result['mean']:.8f}s")
            
            # Linear Search
            def linear_search_test(arr_and_target):
                arr, tgt = arr_and_target
                return LinearSearch.linear_search(arr, tgt)
            
            linear_result = PerformanceBenchmark.benchmark_function(
                linear_search_test, (sorted_array, target)
            )
            results[size]['Linear Search'] = linear_result
            print(f"  Linear Search: {linear_result['mean']:.8f}s")
        
        return results


class DataStructureBenchmark:
    """Benchmarking suite for data structures"""
    
    @staticmethod
    def benchmark_stack_operations(num_operations: int) -> Dict:
        """Benchmark stack operations"""
        print(f"Benchmarking Stack with {num_operations} operations")
        
        def stack_push_pop_test(n):
            stack = Stack()
            # Push operations
            for i in range(n):
                stack.push(i)
            # Pop operations
            while not stack.is_empty():
                stack.pop()
        
        result = PerformanceBenchmark.benchmark_function(
            stack_push_pop_test, num_operations
        )
        print(f"  Stack push/pop {num_operations} items: {result['mean']:.6f}s")
        return result
    
    @staticmethod
    def benchmark_queue_operations(num_operations: int) -> Dict:
        """Benchmark queue operations"""
        print(f"Benchmarking Queue with {num_operations} operations")
        
        def queue_enqueue_dequeue_test(n):
            queue = Queue()
            # Enqueue operations
            for i in range(n):
                queue.enqueue(i)
            # Dequeue operations
            while not queue.is_empty():
                queue.dequeue()
        
        result = PerformanceBenchmark.benchmark_function(
            queue_enqueue_dequeue_test, num_operations
        )
        print(f"  Queue enqueue/dequeue {num_operations} items: {result['mean']:.6f}s")
        return result
    
    @staticmethod
    def benchmark_linked_list_operations(num_operations: int) -> Dict:
        """Benchmark linked list operations"""
        print(f"Benchmarking LinkedList with {num_operations} operations")
        
        def linked_list_test(n):
            ll = LinkedList()
            # Append operations
            for i in range(n):
                ll.append(i)
            # Search operations
            for i in range(0, n, n//10):  # Search for every 10th element
                ll.find(i)
        
        result = PerformanceBenchmark.benchmark_function(
            linked_list_test, num_operations
        )
        print(f"  LinkedList append + search: {result['mean']:.6f}s")
        return result


def generate_performance_report(sorting_results: Dict, searching_results: Dict) -> str:
    """Generate a formatted performance report"""
    report = []
    report.append("# Performance Benchmark Results\n")
    report.append("Generated automatically by PyAlgo benchmark suite.\n")
    
    # Sorting algorithms results
    report.append("## Sorting Algorithms Performance\n")
    report.append("### Random Data Performance (seconds)\n")
    report.append("| Array Size | Merge Sort | Quick Sort | Heap Sort | Bubble Sort |")
    report.append("|------------|------------|------------|-----------|-------------|")
    
    for size in sorted(sorting_results.keys()):
        if 'random' in sorting_results[size]:
            row = f"| {size:,} |"
            for algo in ['Merge Sort', 'Quick Sort', 'Heap Sort', 'Bubble Sort']:
                if algo in sorting_results[size]['random']:
                    result = sorting_results[size]['random'][algo]
                    if 'mean' in result:
                        if result['mean'] == float('inf'):
                            row += " Too slow |"
                        else:
                            row += f" {result['mean']:.6f} |"
                    else:
                        row += " Error |"
                else:
                    row += " N/A |"
            report.append(row)
    
    # Performance insights
    report.append("\n### Performance Insights\n")
    report.append("- **Merge Sort**: Consistent O(n log n) performance across all input types")
    report.append("- **Quick Sort**: Best average performance, but can degrade to O(n²) on sorted data")
    report.append("- **Heap Sort**: Guaranteed O(n log n), good for memory-constrained environments")
    report.append("- **Bubble Sort**: O(n²) complexity, only suitable for small datasets or educational purposes")
    
    # Searching algorithms results
    report.append("\n## Searching Algorithms Performance\n")
    report.append("| Array Size | Binary Search | Linear Search | Speedup Factor |")
    report.append("|------------|---------------|---------------|----------------|")
    
    for size in sorted(searching_results.keys()):
        binary_time = searching_results[size]['Binary Search']['mean']
        linear_time = searching_results[size]['Linear Search']['mean']
        speedup = linear_time / binary_time if binary_time > 0 else 0
        
        report.append(
            f"| {size:,} | {binary_time:.8f} | {linear_time:.8f} | {speedup:.1f}x |"
        )
    
    report.append("\n### Search Performance Insights\n")
    report.append("- **Binary Search**: O(log n) complexity, requires sorted data")
    report.append("- **Linear Search**: O(n) complexity, works on unsorted data")
    report.append("- Binary search shows significant performance advantages as data size increases")
    
    return "\n".join(report)


def main():
    """Run comprehensive performance benchmarks"""
    print("PyAlgo Performance Benchmark Suite")
    print("=" * 50)
    
    # Sorting benchmarks
    print("\n1. Sorting Algorithms Benchmark")
    print("-" * 30)
    sorting_sizes = [100, 500, 1000, 5000]
    sorting_results = SortingBenchmark.benchmark_sorting_algorithms(
        sorting_sizes, ['random']  # Focus on random data for main benchmark
    )
    
    # Searching benchmarks
    print("\n2. Searching Algorithms Benchmark")
    print("-" * 30)
    searching_sizes = [1000, 10000, 100000, 1000000]
    searching_results = SearchingBenchmark.benchmark_search_algorithms(searching_sizes)
    
    # Data structure benchmarks
    print("\n3. Data Structure Operations Benchmark")
    print("-" * 30)
    DataStructureBenchmark.benchmark_stack_operations(10000)
    DataStructureBenchmark.benchmark_queue_operations(10000)
    DataStructureBenchmark.benchmark_linked_list_operations(1000)
    
    # Generate report
    print("\n4. Generating Performance Report")
    print("-" * 30)
    report = generate_performance_report(sorting_results, searching_results)
    
    # Save report to file
    with open('PERFORMANCE_RESULTS.md', 'w') as f:
        f.write(report)
    
    print("Performance report saved to PERFORMANCE_RESULTS.md")
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
