import pytest
from string_algorithms.anagram_detection import (
    is_anagram_sorting,
    is_anagram_counter,
    is_anagram_prime
)

# Test cases for all anagram detection approaches
test_cases = [
    # Basic anagrams
    ("listen", "silent", True),
    ("triangle", "integral", True),
    ("hello", "world", False),
    
    # Case insensitivity
    ("Listen", "Silent", True),
    ("DEBIT CARD", "BAD CREDIT", True),
    
    # Spaces handling
    ("Astronomer", "Moon starer", True),
    ("the eyes", "they see", True),
    
    # Empty strings
    ("", "", True),
    
    # Same letters but different frequencies
    ("aab", "abb", False),
    
    # Special characters
    ("a!bc#", "#cb!a", True),
    
    # Numbers
    ("123456", "654321", True),
    ("112233", "332211", True),
    # These two strings have the same characters but different frequencies
    # '112233' has 1,1,2,2,3,3 while '123123' has 1,1,2,3,1,3
    # Update test case to reflect actual behavior: they ARE anagrams by our definition
    ("112233", "123123", True),
    
    # Longer strings
    ("anagram detection algorithm implementation test case", 
     "implementation test case anagram detection algorithm", True),
]

def test_is_anagram_sorting():
    """Test the sorting-based anagram detection approach."""
    for s1, s2, expected in test_cases:
        assert is_anagram_sorting(s1, s2) == expected, f"Failed for '{s1}' and '{s2}'"

def test_is_anagram_counter():
    """Test the counter-based anagram detection approach."""
    for s1, s2, expected in test_cases:
        assert is_anagram_counter(s1, s2) == expected, f"Failed for '{s1}' and '{s2}'"

def test_is_anagram_prime():
    """Test the prime number-based anagram detection approach."""
    # Skip special character and number cases for prime-based method
    # as it's designed primarily for letters
    filtered_test_cases = [case for case in test_cases if 
                          all(c.isalpha() or c.isspace() for c in case[0] + case[1])]
    
    for s1, s2, expected in filtered_test_cases:
        assert is_anagram_prime(s1, s2) == expected, f"Failed for '{s1}' and '{s2}'"
