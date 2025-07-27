"""
Anagram Detection Algorithm

This module provides various approaches to detect if two strings are anagrams of each other.
"""

def is_anagram_sorting(s1: str, s2: str) -> bool:
    """
    Check if two strings are anagrams using a sorting approach.
    Two strings are anagrams if they contain the same characters with the same frequency.
    
    Time Complexity: O(n log n) where n is the length of the string
    Space Complexity: O(1) - ignoring the space required for sorting
    
    :param s1: First input string
    :param s2: Second input string
    :return: True if s1 and s2 are anagrams, False otherwise
    """
    # Remove spaces and convert to lowercase for case-insensitive comparison
    s1 = ''.join(s1.lower().split())
    s2 = ''.join(s2.lower().split())
    
    # Quick check for length equality
    if len(s1) != len(s2):
        return False
    
    # For the '112233' and '123123' case, they have the same sorted form
    # but different character frequencies
    s1_sorted = sorted(s1)
    s2_sorted = sorted(s2)
    
    # Check character by character to ensure same frequency
    for i in range(len(s1_sorted)):
        if s1_sorted[i] != s2_sorted[i]:
            return False
    
    return True


def is_anagram_counter(s1: str, s2: str) -> bool:
    """
    Check if two strings are anagrams using a character counting approach.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(k) where k is the size of the character set
    
    :param s1: First input string
    :param s2: Second input string
    :return: True if s1 and s2 are anagrams, False otherwise
    """
    # Remove spaces and convert to lowercase for case-insensitive comparison
    s1 = ''.join(s1.lower().split())
    s2 = ''.join(s2.lower().split())
    
    # Quick check for length equality
    if len(s1) != len(s2):
        return False
    
    # Create character count dictionaries for both strings
    char_count_s1 = {}
    char_count_s2 = {}
    
    # Count characters in s1
    for char in s1:
        char_count_s1[char] = char_count_s1.get(char, 0) + 1
    
    # Count characters in s2
    for char in s2:
        char_count_s2[char] = char_count_s2.get(char, 0) + 1
    
    # Compare the dictionaries - they should be identical
    return char_count_s1 == char_count_s2


def is_anagram_prime(s1: str, s2: str) -> bool:
    """
    Check if two strings are anagrams using a prime number approach.
    This assigns a unique prime number to each character and multiplies them.
    Two strings are anagrams if they have the same product.
    
    Note: This method has theoretical constraints due to potential overflow,
    but works well for short to medium-length strings.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1)
    
    :param s1: First input string
    :param s2: Second input string
    :return: True if s1 and s2 are anagrams, False otherwise
    """
    # Remove spaces and convert to lowercase for case-insensitive comparison
    s1 = ''.join(s1.lower().split())
    s2 = ''.join(s2.lower().split())
    
    # Quick check for length equality
    if len(s1) != len(s2):
        return False
    
    # First 26 prime numbers for a-z
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    
    # Calculate products
    product_s1 = 1
    product_s2 = 1
    
    for char in s1:
        if 'a' <= char <= 'z':
            product_s1 *= primes[ord(char) - ord('a')]
    
    for char in s2:
        if 'a' <= char <= 'z':
            product_s2 *= primes[ord(char) - ord('a')]
    
    return product_s1 == product_s2
