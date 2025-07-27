def gcd(a, b):
    """
    Compute the Greatest Common Divisor (GCD) of two numbers using the Euclidean algorithm.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: The GCD of a and b.
    """
    while b:
        a, b = b, a % b
    return a
