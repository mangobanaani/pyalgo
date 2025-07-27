def rabin_karp(text, pattern, prime=101):
    """
    Rabin-Karp pattern matching algorithm.

    :param text: The text to search within.
    :param pattern: The pattern to search for.
    :param prime: A prime number used for hashing.
    :return: The starting index of the first occurrence of the pattern in the text, or -1 if not found.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if n == 0:
        return -1

    base = 256  # Number of characters in the input alphabet
    pattern_hash = 0  # Hash value for the pattern
    text_hash = 0  # Hash value for the text
    h = 1

    # The value of h would be "pow(base, m-1) % prime"
    for _ in range(m - 1):
        h = (h * base) % prime

    # Calculate the hash value of the pattern and the first window of text
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        text_hash = (base * text_hash + ord(text[i])) % prime

    # Slide the pattern over text one by one
    for i in range(n - m + 1):
        # Check the hash values of the current window of text and the pattern
        if pattern_hash == text_hash:
            # Check characters one by one
            if text[i:i + m] == pattern:
                return i

        # Calculate hash value for the next window of text
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime

            # We might get a negative value of text_hash, converting it to positive
            if text_hash < 0:
                text_hash += prime

    return -1
