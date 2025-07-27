def z_algorithm(s):
    """
    Z-Algorithm for pattern matching.

    :param s: The input string (pattern + '$' + text).
    :return: The Z-array where Z[i] is the length of the longest substring starting from s[i] that is also a prefix of s.
    """
    n = len(s)
    z = [0] * n
    l, r, k = 0, 0, 0

    for i in range(1, n):
        if i > r:
            l, r = i, i
            while r < n and s[r] == s[r - l]:
                r += 1
            z[i] = r - l
            r -= 1
        else:
            k = i - l
            if z[k] < r - i + 1:
                z[i] = z[k]
            else:
                l = i
                while r < n and s[r] == s[r - l]:
                    r += 1
                z[i] = r - l
                r -= 1

    return z

def z_algorithm_search(text, pattern):
    """
    Search for all occurrences of a pattern in a text using the Z-Algorithm.

    :param text: The text to search within.
    :param pattern: The pattern to search for.
    :return: A list of starting indices where the pattern is found in the text.
    """
    if not pattern:
        return list(range(len(text)))

    combined = pattern + "$" + text
    z = z_algorithm(combined)
    pattern_length = len(pattern)

    return [i - pattern_length - 1 for i in range(len(z)) if z[i] == pattern_length]
