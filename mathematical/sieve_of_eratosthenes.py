def sieve_of_eratosthenes(n):
    """
    Generate all prime numbers up to n using the Sieve of Eratosthenes.

    Args:
        n (int): The upper limit of numbers to check for primality.

    Returns:
        list: A list of all prime numbers up to n.
    """
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False  # 0 and 1 are not prime numbers

    for i in range(2, int(n**0.5) + 1):
        if primes[i]:
            for j in range(i * i, n + 1, i):
                primes[j] = False

    return [i for i in range(n + 1) if primes[i]]
