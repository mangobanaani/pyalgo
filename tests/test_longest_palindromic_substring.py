import pytest
from string_algorithms.longest_palindromic_substring import longest_palindromic_substring

def test_longest_palindromic_substring():
    assert longest_palindromic_substring("babad") in ["bab", "aba"]  # Both are valid
    assert longest_palindromic_substring("cbbd") == "bb"
    assert longest_palindromic_substring("a") == "a"
    assert longest_palindromic_substring("ac") in ["a", "c"]  # Both are valid
    assert longest_palindromic_substring("") == ""
    assert longest_palindromic_substring("racecar") == "racecar"
    assert longest_palindromic_substring("abacdfgdcaba") == "aba"
    assert longest_palindromic_substring("forgeeksskeegfor") == "geeksskeeg"
    assert longest_palindromic_substring("aabbccddeeffg") in ["aa", "bb", "cc", "dd", "ee", "ff", "g"]
