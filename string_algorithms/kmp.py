import logging

logging.basicConfig(level=logging.INFO)

def compute_lps(pattern):
    """
    Compute the Longest Prefix Suffix (LPS) array for the pattern.
    """
    lps = [0] * len(pattern)
    length = 0  # Length of the previous longest prefix suffix
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def kmp_search(text, pattern):
    """
    Perform KMP (Knuth-Morris-Pratt) pattern matching algorithm.
    """
    lps = compute_lps(pattern)
    i = 0  # Index for text
    j = 0  # Index for pattern
    found = False

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            logging.info(f"Pattern found at index {i - j}")
            found = True
            j = lps[j - 1]

        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    if not found:
        logging.info("Pattern not found in the text.")
