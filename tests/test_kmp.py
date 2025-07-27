import unittest
from string_algorithms.kmp import kmp_search

class TestKMP(unittest.TestCase):
    def test_pattern_found(self):
        text = "ababcabcabababd"
        pattern = "ababd"
        # Capture printed output
        with self.assertLogs() as captured:
            kmp_search(text, pattern)
        self.assertIn("Pattern found at index 10", captured.output[0])

    def test_pattern_not_found(self):
        text = "abcdefgh"
        pattern = "xyz"
        with self.assertLogs() as captured:
            kmp_search(text, pattern)
        self.assertIn("Pattern not found in the text.", captured.output[0])

if __name__ == "__main__":
    unittest.main()
