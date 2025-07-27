import unittest
from string_algorithms.boyer_moore import boyer_moore

class TestBoyerMoore(unittest.TestCase):
    def test_pattern_found(self):
        text = "ABAAABCD"
        pattern = "ABC"
        result = boyer_moore(text, pattern)
        self.assertEqual(result, 4)

    def test_pattern_not_found(self):
        text = "ABAAABCD"
        pattern = "XYZ"
        result = boyer_moore(text, pattern)
        self.assertEqual(result, -1)

    def test_empty_pattern(self):
        text = "ABAAABCD"
        pattern = ""
        result = boyer_moore(text, pattern)
        self.assertEqual(result, 0)

    def test_empty_text(self):
        text = ""
        pattern = "ABC"
        result = boyer_moore(text, pattern)
        self.assertEqual(result, -1)

if __name__ == "__main__":
    unittest.main()
