import pytest
from string_algorithms.string_compression import run_length_encoding, run_length_decoding

def test_run_length_encoding():
    assert run_length_encoding("") == ""
    assert run_length_encoding("A") == "A"  # Single character should not be compressed
    assert run_length_encoding("AABBBCCCC") == "A2B3C4"
    assert run_length_encoding("AAABBBCCC") == "A3B3C3"
    assert run_length_encoding("ABCDEF") == "ABCDEF"  # No compression for non-repeating characters
    assert run_length_encoding("AAAAAAAAAAA") == "A11"
    assert run_length_encoding("ABABABABAB") == "ABABABABAB"  # No compression for alternating pattern

def test_run_length_decoding():
    assert run_length_decoding("") == ""
    assert run_length_decoding("A1") == "A"
    assert run_length_decoding("A2B3C4") == "AABBBCCCC"
    assert run_length_decoding("A3B3C3") == "AAABBBCCC"
    assert run_length_decoding("A11") == "AAAAAAAAAAA"
    
    # Test for characters without explicit counts (implicit count of 1)
    assert run_length_decoding("ABC") == "ABC"
    assert run_length_decoding("A1B1C1") == "ABC"
    
def test_encoding_decoding_roundtrip():
    test_strings = [
        "",
        "A",
        "AABBBCCCC",
        "AAABBBCCC",
        "ABCDEF",
        "AAAAAAAAAAA",
        "ABABABABAB",
        "Hello, World!",
        "aaaaaBBBBBccDDDDeeee"
    ]
    
    for s in test_strings:
        encoded = run_length_encoding(s)
        decoded = run_length_decoding(encoded)
        # For strings that don't benefit from compression, run_length_encoding returns the original
        # So we need to decode the original if that's what was returned
        if encoded == s and any(c.isdigit() for c in s):
            continue
        assert decoded == s, f"Failed round trip for '{s}'"
