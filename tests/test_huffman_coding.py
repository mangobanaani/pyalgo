import unittest
from greedy.huffman_coding import huffman_coding

class TestHuffmanCoding(unittest.TestCase):
    def test_simple_string(self):
        s = "aaabbc"
        codes = huffman_coding(s)
        self.assertEqual(set(codes.keys()), set("abc"))
        self.assertTrue(all(len(code) > 0 for code in codes.values()))

    def test_empty_string(self):
        s = ""
        codes = huffman_coding(s)
        self.assertEqual(codes, {})

    def test_single_character(self):
        s = "aaaa"
        codes = huffman_coding(s)
        self.assertEqual(codes, {"a": ""})

if __name__ == "__main__":
    unittest.main()
