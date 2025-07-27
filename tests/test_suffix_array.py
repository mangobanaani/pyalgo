import pytest
from string_algorithms.suffix_array import (
    build_suffix_array_naive,
    build_suffix_array_improved,
    longest_common_prefix,
    build_lcp_array
)

def test_build_suffix_array_naive():
    """Test the naive suffix array construction."""
    assert build_suffix_array_naive("banana") == [5, 3, 1, 0, 4, 2]
    assert build_suffix_array_naive("abracadabra") == [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]
    assert build_suffix_array_naive("mississippi") == [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
    assert build_suffix_array_naive("") == []
    assert build_suffix_array_naive("a") == [0]

def test_build_suffix_array_improved():
    """Test the improved suffix array construction."""
    assert build_suffix_array_improved("banana") == [5, 3, 1, 0, 4, 2]
    assert build_suffix_array_improved("abracadabra") == [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]
    assert build_suffix_array_improved("mississippi") == [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
    assert build_suffix_array_improved("") == []
    assert build_suffix_array_improved("a") == [0]

def test_longest_common_prefix():
    """Test the longest common prefix calculation."""
    s = "banana"
    assert longest_common_prefix(s, 0, 2) == 0  # 'banana' and 'nana'
    assert longest_common_prefix(s, 1, 3) == 1  # 'anana' and 'ana'
    assert longest_common_prefix(s, 0, 0) == 6  # 'banana' and 'banana'

    s = "abracadabra"
    assert longest_common_prefix(s, 0, 7) == 4  # 'abracadabra' and 'abra'
    assert longest_common_prefix(s, 0, 10) == 0  # 'abracadabra' and 'a'

def test_build_lcp_array():
    """Test the LCP array construction."""
    s = "banana"
    sa = build_suffix_array_improved(s)
    lcp = build_lcp_array(s, sa)
    assert lcp == [0, 1, 3, 0, 0, 2]

    s = "abracadabra"
    sa = build_suffix_array_improved(s)
    lcp = build_lcp_array(s, sa)
    assert lcp == [0, 1, 4, 1, 1, 0, 3, 0, 0, 0, 2]

def test_empty_and_single_char():
    """Test empty string and single character cases."""
    # Empty string
    s = ""
    assert build_suffix_array_naive(s) == []
    assert build_suffix_array_improved(s) == []
    
    # Single character
    s = "a"
    assert build_suffix_array_naive(s) == [0]
    assert build_suffix_array_improved(s) == [0]
    lcp = build_lcp_array(s, [0])
    assert lcp == [0]

def test_algorithms_equivalence():
    """Test that both suffix array algorithms produce the same result."""
    test_strings = [
        "banana",
        "abracadabra",
        "mississippi",
        "algorithmsareawesome",
        "aaaaaa",
        "abcdefghijklmnopqrstuvwxyz"
    ]
    
    for s in test_strings:
        naive_sa = build_suffix_array_naive(s)
        improved_sa = build_suffix_array_improved(s)
        assert naive_sa == improved_sa, f"Failed for string '{s}'"
