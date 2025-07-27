import pytest
from string_algorithms.palindrome_checking import (
    is_palindrome_simple,
    is_palindrome_two_pointers,
    is_palindrome_recursive,
    is_palindrome_in_place
)

# Test cases for all palindrome checking approaches
test_cases = [
    # Simple palindromes
    ("racecar", True),
    ("radar", True),
    ("level", True),
    ("hello", False),
    
    # Case insensitivity
    ("Racecar", True),
    ("RaceCar", True),
    
    # Spaces and punctuation
    ("A man, a plan, a canal: Panama", True),
    ("No 'x' in Nixon", True),
    ("Was it a car or a cat I saw?", True),
    
    # Empty and single character strings
    ("", True),
    ("a", True),
    
    # Numeric palindromes
    ("12321", True),
    ("123", False),
    
    # Mixed content
    ("A1B2C3C2B1A", True),
    ("A1B2C3D4", False),
    
    # Longer palindromes
    ("amanaplanacanalpanama", True),
    ("thequickbrownfoxjumpsoverthelazydog", False),
]

def test_is_palindrome_simple():
    """Test the simple palindrome checking approach."""
    for s, expected in test_cases:
        assert is_palindrome_simple(s) == expected, f"Failed for '{s}'"

def test_is_palindrome_two_pointers():
    """Test the two-pointer technique palindrome checking approach."""
    for s, expected in test_cases:
        assert is_palindrome_two_pointers(s) == expected, f"Failed for '{s}'"

def test_is_palindrome_recursive():
    """Test the recursive palindrome checking approach."""
    for s, expected in test_cases:
        assert is_palindrome_recursive(s) == expected, f"Failed for '{s}'"

def test_is_palindrome_in_place():
    """Test the in-place palindrome checking approach."""
    for s, expected in test_cases:
        assert is_palindrome_in_place(s) == expected, f"Failed for '{s}'"
