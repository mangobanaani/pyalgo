import unittest
from string_algorithms.trie import Trie

class TestTrie(unittest.TestCase):
    def setUp(self):
        self.trie = Trie()

    def test_insert_and_search(self):
        self.trie.insert("apple")
        self.assertTrue(self.trie.search("apple"))  # Exact match
        self.assertFalse(self.trie.search("app"))   # Prefix only

    def test_starts_with(self):
        self.trie.insert("apple")
        self.assertTrue(self.trie.starts_with("app"))  # Prefix match
        self.assertFalse(self.trie.starts_with("apl")) # Non-matching prefix

if __name__ == "__main__":
    unittest.main()
