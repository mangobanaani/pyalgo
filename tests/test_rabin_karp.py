import unittest
from string_algorithms.rabin_karp import rabin_karp

class TestRabinKarp(unittest.TestCase):
    def test_pattern_found(self):
        text = "GEEKS FOR GEEKS"
        pattern = "GEEK"
        result = rabin_karp(text, pattern)
        self.assertEqual(result, 0)

    def test_pattern_not_found(self):
        text = "GEEKS FOR GEEKS"
        pattern = "QUIZ"
        result = rabin_karp(text, pattern)
        self.assertEqual(result, -1)

    def test_empty_pattern(self):
        text = "GEEKS FOR GEEKS"
        pattern = ""
        result = rabin_karp(text, pattern)
        self.assertEqual(result, 0)

    def test_empty_text(self):
        text = ""
        pattern = "GEEK"
        result = rabin_karp(text, pattern)
        self.assertEqual(result, -1)

if __name__ == "__main__":
    unittest.main()
