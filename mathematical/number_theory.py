"""
Number Theory algorithms.

This module implements fundamental number theory algorithms including
primality testing, factorization, and modular arithmetic.
"""

from typing import List, Tuple, Optional
import math


def gcd(a: int, b: int) -> int:
    """
    Calculate Greatest Common Divisor using Euclidean algorithm.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Greatest common divisor of a and b
        
    Time Complexity: O(log(min(a,b)))
    Space Complexity: O(1)
    """
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """
    Calculate Least Common Multiple.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Least common multiple of a and b
        
    Time Complexity: O(log(min(a,b)))
    Space Complexity: O(1)
    """
    return abs(a * b) // gcd(a, b) if a and b else 0


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.
    
    Finds gcd(a,b) and coefficients x,y such that ax + by = gcd(a,b).
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        Tuple (gcd, x, y) where ax + by = gcd
        
    Time Complexity: O(log(min(a,b)))
    Space Complexity: O(1)
    """
    if a == 0:
        return b, 0, 1
    
    gcd_val, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    
    return gcd_val, x, y


def mod_inverse(a: int, m: int) -> Optional[int]:
    """
    Calculate modular inverse of a modulo m.
    
    Args:
        a: Integer to find inverse of
        m: Modulus
        
    Returns:
        Modular inverse if it exists, None otherwise
        
    Time Complexity: O(log(min(a,m)))
    Space Complexity: O(1)
    """
    gcd_val, x, _ = extended_gcd(a, m)
    
    if gcd_val != 1:
        return None  # Inverse doesn't exist
    
    return (x % m + m) % m


def is_prime(n: int) -> bool:
    """
    Check if a number is prime using trial division.
    
    Args:
        n: Number to check
        
    Returns:
        True if n is prime, False otherwise
        
    Time Complexity: O(√n)
    Space Complexity: O(1)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    
    return True


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Find all prime numbers up to limit using Sieve of Eratosthenes.
    
    Args:
        limit: Upper bound (inclusive)
        
    Returns:
        List of all prime numbers <= limit
        
    Time Complexity: O(n log log n)
    Space Complexity: O(n)
    """
    if limit < 2:
        return []
    
    # Create boolean array and set all entries as True
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    
    p = 2
    while p * p <= limit:
        if is_prime_arr[p]:
            # Mark all multiples of p as composite
            for i in range(p * p, limit + 1, p):
                is_prime_arr[i] = False
        p += 1
    
    # Collect all prime numbers
    return [i for i in range(2, limit + 1) if is_prime_arr[i]]


def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """
    Find prime factorization of a number.
    
    Args:
        n: Number to factorize
        
    Returns:
        List of (prime, exponent) tuples
        
    Time Complexity: O(√n)
    Space Complexity: O(log n)
    """
    if n <= 1:
        return []
    
    factors = []
    
    # Check for factor 2
    if n % 2 == 0:
        count = 0
        while n % 2 == 0:
            count += 1
            n //= 2
        factors.append((2, count))
    
    # Check for odd factors
    i = 3
    while i * i <= n:
        if n % i == 0:
            count = 0
            while n % i == 0:
                count += 1
                n //= i
            factors.append((i, count))
        i += 2
    
    # If n is still > 1, then it's a prime
    if n > 1:
        factors.append((n, 1))
    
    return factors


def fast_power(base: int, exp: int, mod: int = None) -> int:
    """
    Calculate base^exp efficiently using binary exponentiation.
    
    Args:
        base: Base number
        exp: Exponent
        mod: Optional modulus
        
    Returns:
        base^exp (mod mod if specified)
        
    Time Complexity: O(log exp)
    Space Complexity: O(1)
    """
    if exp == 0:
        return 1
    
    result = 1
    base = base % mod if mod else base
    
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod if mod else result * base
        
        exp = exp >> 1  # Divide by 2
        base = (base * base) % mod if mod else base * base
    
    return result


def euler_totient(n: int) -> int:
    """
    Calculate Euler's totient function φ(n).
    
    φ(n) counts positive integers up to n that are coprime to n.
    
    Args:
        n: Input number
        
    Returns:
        Value of φ(n)
        
    Time Complexity: O(√n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return 0 if n <= 0 else 1
    
    result = n
    
    # Check all prime factors
    p = 2
    while p * p <= n:
        if n % p == 0:
            # Remove all factors of p
            while n % p == 0:
                n //= p
            # Apply formula: φ(n) = n * (1 - 1/p)
            result -= result // p
        p += 1
    
    # If n is still > 1, then it's a prime
    if n > 1:
        result -= result // n
    
    return result


def miller_rabin(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin primality test (probabilistic).
    
    Args:
        n: Number to test
        k: Number of rounds (higher = more accurate)
        
    Returns:
        True if n is probably prime, False if composite
        
    Time Complexity: O(k log³ n)
    Space Complexity: O(1)
    """
    import random
    
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as d * 2^r
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Perform k rounds of testing
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = fast_power(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = fast_power(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True


def carmichael_function(n: int) -> int:
    """
    Calculate Carmichael function λ(n).
    
    λ(n) is the smallest positive integer m such that a^m ≡ 1 (mod n)
    for every integer a coprime to n.
    
    Args:
        n: Input number
        
    Returns:
        Value of λ(n)
        
    Time Complexity: O(√n)
    Space Complexity: O(log n)
    """
    if n <= 1:
        return 1
    
    factors = prime_factorization(n)
    carmichael_values = []
    
    for prime, exp in factors:
        if prime == 2 and exp >= 3:
            # Special case for powers of 2
            carmichael_values.append(2**(exp - 2))
        elif prime == 2:
            carmichael_values.append(2**(exp - 1))
        else:
            # For odd primes: λ(p^k) = φ(p^k) = p^(k-1) * (p-1)
            carmichael_values.append((prime**(exp - 1)) * (prime - 1))
    
    # λ(n) = lcm of all λ(p^k)
    result = carmichael_values[0]
    for val in carmichael_values[1:]:
        result = lcm(result, val)
    
    return result


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> Optional[int]:
    """
    Solve system of congruences using Chinese Remainder Theorem.
    
    Solves: x ≡ r₁ (mod m₁), x ≡ r₂ (mod m₂), ..., x ≡ rₖ (mod mₖ)
    
    Args:
        remainders: List of remainders [r₁, r₂, ..., rₖ]
        moduli: List of moduli [m₁, m₂, ..., mₖ] (must be pairwise coprime)
        
    Returns:
        Solution x if it exists, None otherwise
        
    Time Complexity: O(n log max(moduli))
    Space Complexity: O(1)
    """
    if len(remainders) != len(moduli):
        return None
    
    if not remainders:
        return 0
    
    # Check if moduli are pairwise coprime
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                return None  # Not pairwise coprime
    
    # Calculate product of all moduli
    M = 1
    for m in moduli:
        M *= m
    
    x = 0
    for i in range(len(remainders)):
        Mi = M // moduli[i]
        yi = mod_inverse(Mi, moduli[i])
        
        if yi is None:
            return None  # Inverse doesn't exist
        
        x = (x + remainders[i] * Mi * yi) % M
    
    return x
