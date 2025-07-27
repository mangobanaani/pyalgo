def longest_common_substring(s1, s2):
    """
    Find the longest common substring between two strings.

    :param s1: First string.
    :param s2: Second string.
    :return: The longest common substring and its length.
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_length = 0
    end_index = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i

    return s1[end_index - max_length:end_index], max_length
