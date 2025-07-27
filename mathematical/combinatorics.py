"""
Combinatorics algorithms and functions.

This module implements combinatorial functions, sequence generation,
and counting algorithms.
"""

from typing import List, Iterator, Optional
import math


def factorial(n: int) -> int:
    """
    Calculate factorial of n.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n! = n * (n-1) * ... * 2 * 1
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result


def factorial_iterative_memoized(n: int, memo: dict = None) -> int:
    """
    Calculate factorial with memoization.
    
    Args:
        n: Non-negative integer
        memo: Memoization dictionary
        
    Returns:
        n!
        
    Time Complexity: O(n) first call, O(1) subsequent calls for same n
    Space Complexity: O(n)
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        memo[n] = 1
        return 1
    
    result = n * factorial_iterative_memoized(n - 1, memo)
    memo[n] = result
    return result


def permutations(n: int, r: int = None) -> int:
    """
    Calculate number of permutations P(n,r) = n!/(n-r)!
    
    Args:
        n: Total number of items
        r: Number of items to arrange (default: n)
        
    Returns:
        Number of permutations
        
    Time Complexity: O(r)
    Space Complexity: O(1)
    """
    if r is None:
        r = n
    
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    
    if r > n:
        return 0
    
    if r == 0:
        return 1
    
    result = 1
    for i in range(n, n - r, -1):
        result *= i
    
    return result


def combinations(n: int, r: int) -> int:
    """
    Calculate number of combinations C(n,r) = n!/(r!(n-r)!)
    
    Args:
        n: Total number of items
        r: Number of items to choose
        
    Returns:
        Number of combinations
        
    Time Complexity: O(min(r, n-r))
    Space Complexity: O(1)
    """
    if n < 0 or r < 0:
        raise ValueError("n and r must be non-negative")
    
    if r > n:
        return 0
    
    if r == 0 or r == n:
        return 1
    
    # Use symmetry: C(n,r) = C(n,n-r)
    r = min(r, n - r)
    
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    
    return result


def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence (0-indexed)
        
    Returns:
        nth Fibonacci number
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def fibonacci_sequence(n: int) -> List[int]:
    """
    Generate first n Fibonacci numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate
        
    Returns:
        List of first n Fibonacci numbers
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n <= 0:
        return []
    
    if n == 1:
        return [0]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib


def lucas_numbers(n: int) -> int:
    """
    Calculate nth Lucas number.
    
    Lucas sequence: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2)
    
    Args:
        n: Position in Lucas sequence
        
    Returns:
        nth Lucas number
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n == 0:
        return 2
    if n == 1:
        return 1
    
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


def catalan_number(n: int) -> int:
    """
    Calculate nth Catalan number.
    
    C(n) = (1/(n+1)) * C(2n,n) = (2n)! / ((n+1)! * n!)
    
    Args:
        n: Position in Catalan sequence
        
    Returns:
        nth Catalan number
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n == 0:
        return 1
    
    # Use the recurrence relation: C(n) = sum(C(i) * C(n-1-i)) for i=0 to n-1
    # But we'll use the direct formula for efficiency
    result = 1
    for i in range(n):
        result = result * (2 * n - i) // (i + 1)
    
    return result // (n + 1)


def stirling_second_kind(n: int, k: int) -> int:
    """
    Calculate Stirling number of the second kind S(n,k).
    
    S(n,k) counts the number of ways to partition n objects into k non-empty subsets.
    
    Args:
        n: Number of objects
        k: Number of subsets
        
    Returns:
        Stirling number S(n,k)
        
    Time Complexity: O(nk)
    Space Complexity: O(nk)
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    
    if k == 0:
        return 1 if n == 0 else 0
    
    if k > n:
        return 0
    
    if k == 1 or k == n:
        return 1
    
    # Use dynamic programming
    S = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    
    # Base cases
    S[0][0] = 1
    for i in range(1, n + 1):
        S[i][0] = 0
        S[i][1] = 1
        if i <= k:
            S[i][i] = 1
    
    # Fill the table
    for i in range(2, n + 1):
        for j in range(2, min(i, k) + 1):
            S[i][j] = j * S[i-1][j] + S[i-1][j-1]
    
    return S[n][k]


def bell_number(n: int) -> int:
    """
    Calculate nth Bell number.
    
    B(n) counts the number of ways to partition n objects.
    B(n) = sum(S(n,k)) for k=0 to n
    
    Args:
        n: Number of objects
        
    Returns:
        nth Bell number
        
    Time Complexity: O(n²)
    Space Complexity: O(n²)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n == 0:
        return 1
    
    # Use Bell triangle
    bell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    bell[0][0] = 1
    
    for i in range(1, n + 1):
        # First element of each row is same as last element of previous row
        bell[i][0] = bell[i-1][i-1]
        
        # Fill the rest of the row
        for j in range(1, i + 1):
            bell[i][j] = bell[i-1][j-1] + bell[i][j-1]
    
    return bell[n][0]


def partition_function(n: int, max_value: int = None) -> int:
    """
    Calculate number of integer partitions of n.
    
    Args:
        n: Number to partition
        max_value: Maximum value allowed in partition (default: n)
        
    Returns:
        Number of partitions
        
    Time Complexity: O(n * max_value)
    Space Complexity: O(n)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if max_value is None:
        max_value = n
    
    if n == 0:
        return 1
    
    # Dynamic programming approach
    dp = [0] * (n + 1)
    dp[0] = 1
    
    for i in range(1, min(max_value, n) + 1):
        for j in range(i, n + 1):
            dp[j] += dp[j - i]
    
    return dp[n]


def generate_permutations(items: List) -> Iterator[List]:
    """
    Generate all permutations of given items.
    
    Args:
        items: List of items to permute
        
    Yields:
        Each permutation as a list
        
    Time Complexity: O(n! * n)
    Space Complexity: O(n)
    """
    if len(items) <= 1:
        yield items[:]
        return
    
    for i in range(len(items)):
        rest = items[:i] + items[i+1:]
        for perm in generate_permutations(rest):
            yield [items[i]] + perm


def generate_combinations(items: List, r: int) -> Iterator[List]:
    """
    Generate all combinations of r items from given list.
    
    Args:
        items: List of items
        r: Number of items to choose
        
    Yields:
        Each combination as a list
        
    Time Complexity: O(C(n,r) * r)
    Space Complexity: O(r)
    """
    if r == 0:
        yield []
        return
    
    if r > len(items):
        return
    
    for i in range(len(items)):
        item = items[i]
        rest = items[i+1:]
        
        for comb in generate_combinations(rest, r - 1):
            yield [item] + comb


def pascals_triangle(n: int) -> List[List[int]]:
    """
    Generate first n rows of Pascal's triangle.
    
    Args:
        n: Number of rows
        
    Returns:
        List of rows, where each row is a list of binomial coefficients
        
    Time Complexity: O(n²)
    Space Complexity: O(n²)
    """
    if n <= 0:
        return []
    
    triangle = []
    
    for i in range(n):
        row = [1]  # First element is always 1
        
        # Calculate middle elements
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        
        # Last element is always 1 (except for row 0)
        if i > 0:
            row.append(1)
        
        triangle.append(row)
    
    return triangle


def multinomial_coefficient(n: int, groups: List[int]) -> int:
    """
    Calculate multinomial coefficient.
    
    Coefficient = n! / (k1! * k2! * ... * km!)
    where k1 + k2 + ... + km = n
    
    Args:
        n: Total number of objects
        groups: Sizes of each group
        
    Returns:
        Multinomial coefficient
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if sum(groups) != n:
        raise ValueError("Sum of group sizes must equal n")
    
    if any(k < 0 for k in groups):
        raise ValueError("All group sizes must be non-negative")
    
    # Calculate n! / (k1! * k2! * ... * km!)
    result = factorial(n)
    
    for k in groups:
        result //= factorial(k)
    
    return result


def derangements(n: int) -> int:
    """
    Calculate number of derangements of n objects.
    
    A derangement is a permutation where no element appears in its original position.
    
    Args:
        n: Number of objects
        
    Returns:
        Number of derangements
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n == 0:
        return 1
    if n == 1:
        return 0
    if n == 2:
        return 1
    
    # Use recurrence: D(n) = (n-1) * (D(n-1) + D(n-2))
    d_prev_prev = 1  # D(0)
    d_prev = 0       # D(1)
    
    for i in range(2, n + 1):
        d_curr = (i - 1) * (d_prev + d_prev_prev)
        d_prev_prev, d_prev = d_prev, d_curr
    
    return d_prev
