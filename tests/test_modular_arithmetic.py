import pytest
from mathematical.modular_arithmetic import (
    modular_exponentiation,
    modular_inverse,
    extended_gcd
)

def test_modular_exponentiation():
    """Test the modular exponentiation function."""
    # Basic tests
    assert modular_exponentiation(2, 3, 5) == 3  # 2^3 = 8, 8 % 5 = 3
    assert modular_exponentiation(3, 4, 7) == 4  # 3^4 = 81, 81 % 7 = 4
    assert modular_exponentiation(10, 9, 6) == 4  # 10^9 % 6 = 4
    
    # Edge cases
    assert modular_exponentiation(2, 0, 10) == 1  # 2^0 = 1, 1 % 10 = 1
    assert modular_exponentiation(0, 5, 7) == 0  # 0^5 = 0, 0 % 7 = 0
    assert modular_exponentiation(1, 1000000, 11) == 1  # 1^1000000 = 1, 1 % 11 = 1
    
    # Large exponents (which would overflow without modular arithmetic)
    assert modular_exponentiation(2, 100, 1000) == 976  # 2^100 % 1000 = 976
    assert modular_exponentiation(7, 256, 13) == 9  # 7^256 % 13 = 9

def test_extended_gcd():
    """Test the extended GCD function."""
    # Test cases with known results
    assert extended_gcd(35, 15) == (5, 1, -2)  # 5 = 35*1 + 15*(-2)
    assert extended_gcd(12, 8) == (4, 1, -1)  # 4 = 12*1 + 8*(-1)
    assert extended_gcd(31, 37) == (1, 6, -5)  # 1 = 31*6 + 37*(-5)
    assert extended_gcd(0, 5) == (5, 0, 1)  # 5 = 0*0 + 5*1
    assert extended_gcd(5, 0) == (5, 1, 0)  # 5 = 5*1 + 0*0
    
    # Verify the Bézout's identity: a*x + b*y = gcd(a,b)
    for a, b in [(35, 15), (12, 8), (31, 37), (17, 13), (100, 7)]:
        gcd, x, y = extended_gcd(a, b)
        assert a * x + b * y == gcd, f"Failed for a={a}, b={b}"

def test_modular_inverse():
    """Test the modular inverse function."""
    # Test cases with known results
    assert modular_inverse(3, 11) == 4  # (3 * 4) % 11 = 1
    assert modular_inverse(10, 17) == 12  # (10 * 12) % 17 = 1
    assert modular_inverse(7, 20) == 3  # (7 * 3) % 20 = 1
    
    # Verify the definition of modular inverse: (a * a_inv) % m = 1
    for a, m in [(3, 11), (10, 17), (7, 20), (5, 24), (2, 7)]:
        a_inv = modular_inverse(a, m)
        assert (a * a_inv) % m == 1, f"Failed for a={a}, m={m}"
    
    # Test when modular inverse doesn't exist (a and m are not coprime)
    with pytest.raises(ValueError):
        modular_inverse(4, 8)  # gcd(4, 8) = 4, not 1
    with pytest.raises(ValueError):
        modular_inverse(6, 9)  # gcd(6, 9) = 3, not 1

def test_negative_exponent():
    """Test modular exponentiation with negative exponent."""
    # For negative exponents, we should compute the modular inverse
    # and then raise it to the positive exponent
    assert modular_exponentiation(2, -3, 5) == 3  # 2^-3 mod 5 = (2^-1)^3 mod 5 = 3^3 mod 5 = 2
    
    # More complex case
    assert modular_exponentiation(3, -2, 7) == 4  # 3^-2 mod 7 = (3^-1)^2 mod 7 = 5^2 mod 7 = 4
