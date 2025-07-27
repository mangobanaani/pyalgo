import unittest
from string_algorithms.z_algorithm import z_algorithm_search

class TestZAlgorithm(unittest.TestCase):
    def test_pattern_found(self):
        text = "ababcababc"
        pattern = "abc"
        result = z_algorithm_search(text, pattern)
        self.assertEqual(result, [2, 7])

    def test_pattern_not_found(self):
        text = "ababcababc"
        pattern = "xyz"
        result = z_algorithm_search(text, pattern)
        self.assertEqual(result, [])

    def test_empty_pattern(self):
        text = "ababcababc"
        pattern = ""
        result = z_algorithm_search(text, pattern)
        self.assertEqual(result, list(range(len(text))))

    def test_empty_text(self):
        text = ""
        pattern = "abc"
        result = z_algorithm_search(text, pattern)
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
