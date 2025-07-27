def boyer_moore(text, pattern):
    """
    Boyer-Moore pattern matching algorithm.

    :param text: The text to search within.
    :param pattern: The pattern to search for.
    :return: The starting index of the first occurrence of the pattern in the text, or -1 if not found.
    """
    def preprocess_bad_character(pattern):
        bad_char = {}
        for i, char in enumerate(pattern):
            bad_char[char] = i
        return bad_char

    def preprocess_good_suffix(pattern):
        m = len(pattern)
        good_suffix = [0] * m
        border = 0
        for i in range(m - 1, -1, -1):
            if pattern[i:] == pattern[:m - i]:
                border = m - i
            good_suffix[i] = border
        return good_suffix

    n, m = len(text), len(pattern)
    if m == 0:
        return 0

    bad_char = preprocess_bad_character(pattern)
    good_suffix = preprocess_good_suffix(pattern)

    s = 0  # Shift of the pattern with respect to text
    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            return s
        else:
            bad_char_shift = j - bad_char.get(text[s + j], -1)
            good_suffix_shift = good_suffix[j] if j < m - 1 else 1
            s += max(bad_char_shift, good_suffix_shift)

    return -1
