import unittest
from string_algorithms.longest_common_substring import longest_common_substring

class TestLongestCommonSubstring(unittest.TestCase):
    def test_common_substring(self):
        s1 = "abcdef"
        s2 = "zcdemf"
        result, length = longest_common_substring(s1, s2)
        self.assertEqual(result, "cde")
        self.assertEqual(length, 3)

    def test_no_common_substring(self):
        s1 = "abc"
        s2 = "xyz"
        result, length = longest_common_substring(s1, s2)
        self.assertEqual(result, "")
        self.assertEqual(length, 0)

    def test_empty_strings(self):
        s1 = ""
        s2 = ""
        result, length = longest_common_substring(s1, s2)
        self.assertEqual(result, "")
        self.assertEqual(length, 0)

if __name__ == "__main__":
    unittest.main()
