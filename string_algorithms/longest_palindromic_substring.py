def longest_palindromic_substring(s: str) -> str:
    """
    Finds the longest palindromic substring in the given string using Manacher's algorithm.

    :param s: Input string
    :return: Longest palindromic substring
    """
    if not s:
        return ""

    # Transform the string to avoid even/odd length issues
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n  # Array to store the radius of the palindrome at each position
    center = 0
    right = 0

    for i in range(n):
        mirror = 2 * center - i  # Mirror position of i with respect to center

        if i < right:
            p[i] = min(right - i, p[mirror])

        # Expand around the current center
        while i + p[i] + 1 < n and i - p[i] - 1 >= 0 and t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1

        # Update the center and right boundary if the palindrome expands beyond the current right boundary
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find the maximum length palindrome
    max_len = max(p)
    center_index = p.index(max_len)

    # Extract the original substring from the transformed string
    start = (center_index - max_len) // 2
    return s[start:start + max_len]
