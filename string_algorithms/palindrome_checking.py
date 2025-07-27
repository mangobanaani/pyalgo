"""
Palindrome Checking Algorithm

This module provides various approaches to efficiently check if a string is a palindrome.
A palindrome is a string that reads the same forward and backward.
"""

def is_palindrome_simple(s: str) -> bool:
    """
    Checks if a string is a palindrome using the simplest approach.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) due to string cleaning operations
    
    :param s: Input string
    :return: True if s is a palindrome, False otherwise
    """
    # Remove spaces, punctuation and convert to lowercase
    s = ''.join(c.lower() for c in s if c.isalnum())
    
    # Compare the string with its reverse
    return s == s[::-1]


def is_palindrome_two_pointers(s: str) -> bool:
    """
    Checks if a string is a palindrome using a two-pointer technique.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) due to string cleaning operations
    
    :param s: Input string
    :return: True if s is a palindrome, False otherwise
    """
    # Remove spaces, punctuation and convert to lowercase
    s = ''.join(c.lower() for c in s if c.isalnum())
    
    # Use two pointers from both ends
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True


def is_palindrome_recursive(s: str) -> bool:
    """
    Checks if a string is a palindrome using recursion.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(n) due to recursion stack and string cleaning
    
    :param s: Input string
    :return: True if s is a palindrome, False otherwise
    """
    # Helper function to check palindrome recursively
    def _is_palindrome(s, left, right):
        # Base case: if we've reached the middle or crossed over
        if left >= right:
            return True
        
        # If characters at both ends match, check the inner substring
        if s[left] == s[right]:
            return _is_palindrome(s, left + 1, right - 1)
        
        # If characters don't match, it's not a palindrome
        return False
    
    # Clean the string
    s = ''.join(c.lower() for c in s if c.isalnum())
    
    # Check if the cleaned string is a palindrome
    return _is_palindrome(s, 0, len(s) - 1)


def is_palindrome_in_place(s: str) -> bool:
    """
    Checks if a string is a palindrome using in-place character comparison.
    
    Time Complexity: O(n) where n is the length of the string
    Space Complexity: O(1) as we use constant extra space
    
    :param s: Input string
    :return: True if s is a palindrome, False otherwise
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters from left
        while left < right and not s[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric characters from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
