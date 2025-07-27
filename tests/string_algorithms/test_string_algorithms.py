import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from string_algorithms import Trie, KMPPatternMatcher, StringUtils


class TestStringAlgorithms(unittest.TestCase):
    """Test suite for string algorithms"""
    
    def test_trie_basic_operations(self):
        """Test basic Trie operations"""
        trie = Trie()
        
        # Test empty trie
        self.assertTrue(trie.is_empty())
        self.assertEqual(trie.count_words(), 0)
        
        # Test insertion
        words = ["cat", "car", "card", "care", "careful", "cats", "dog"]
        for word in words:
            trie.insert(word)
        
        self.assertFalse(trie.is_empty())
        self.assertEqual(trie.count_words(), len(words))
        
        # Test search
        for word in words:
            self.assertTrue(trie.search(word))
        
        self.assertFalse(trie.search("ca"))
        self.assertFalse(trie.search("cars"))
        self.assertFalse(trie.search("caterpillar"))
    
    def test_trie_prefix_operations(self):
        """Test Trie prefix operations"""
        trie = Trie()
        words = ["cat", "car", "card", "care", "careful"]
        
        for word in words:
            trie.insert(word)
        
        # Test starts_with
        self.assertTrue(trie.starts_with("ca"))
        self.assertTrue(trie.starts_with("car"))
        self.assertTrue(trie.starts_with("care"))
        self.assertFalse(trie.starts_with("dog"))
        
        # Test get_words_with_prefix
        car_words = trie.get_words_with_prefix("car")
        expected_car_words = ["car", "card", "care", "careful"]
        self.assertEqual(sorted(car_words), sorted(expected_car_words))
        
        care_words = trie.get_words_with_prefix("care")
        expected_care_words = ["care", "careful"]
        self.assertEqual(sorted(care_words), sorted(expected_care_words))
    
    def test_trie_deletion(self):
        """Test Trie deletion"""
        trie = Trie()
        words = ["cat", "car", "card"]
        
        for word in words:
            trie.insert(word)
        
        # Delete a word
        self.assertTrue(trie.delete("car"))
        self.assertFalse(trie.search("car"))
        self.assertTrue(trie.search("cat"))
        self.assertTrue(trie.search("card"))
        
        # Try to delete non-existent word
        self.assertFalse(trie.delete("dog"))
        
        # Word count should be updated
        self.assertEqual(trie.count_words(), 2)
    
    def test_kmp_pattern_matching(self):
        """Test KMP pattern matching"""
        text = "ABABDABACDABABCABCABCABCABC"
        pattern = "ABABCABCABCABC"
        
        # Test search
        occurrences = KMPPatternMatcher.search(text, pattern)
        self.assertEqual(occurrences, [10])  # Pattern starts at index 10
        
        # Test find_first
        first = KMPPatternMatcher.find_first(text, pattern)
        self.assertEqual(first, 10)
        
        # Test count_occurrences
        count = KMPPatternMatcher.count_occurrences(text, pattern)
        self.assertEqual(count, 1)
        
        # Test multiple occurrences
        text2 = "AAAAAAA"
        pattern2 = "AAA"
        occurrences2 = KMPPatternMatcher.search(text2, pattern2)
        self.assertEqual(occurrences2, [0, 1, 2, 3, 4])
    
    def test_kmp_edge_cases(self):
        """Test KMP edge cases"""
        # Empty pattern
        self.assertEqual(KMPPatternMatcher.search("hello", ""), [])
        
        # Empty text
        self.assertEqual(KMPPatternMatcher.search("", "hello"), [])
        
        # Pattern longer than text
        self.assertEqual(KMPPatternMatcher.search("hi", "hello"), [])
        
        # No match
        self.assertEqual(KMPPatternMatcher.search("hello", "xyz"), [])
        
        # Single character
        self.assertEqual(KMPPatternMatcher.search("aaa", "a"), [0, 1, 2])
    
    def test_string_utils_palindrome(self):
        """Test palindrome detection"""
        # Basic palindromes
        self.assertTrue(StringUtils.is_palindrome("racecar"))
        self.assertTrue(StringUtils.is_palindrome("A man a plan a canal Panama"))
        self.assertTrue(StringUtils.is_palindrome("Was it a car or a cat I saw"))
        
        # Non-palindromes
        self.assertFalse(StringUtils.is_palindrome("hello"))
        self.assertFalse(StringUtils.is_palindrome("race a car"))
        
        # Case sensitivity
        self.assertTrue(StringUtils.is_palindrome("Aa", ignore_case=True))
        self.assertFalse(StringUtils.is_palindrome("Aa", ignore_case=False))
        
        # With spaces
        self.assertTrue(StringUtils.is_palindrome("a b a", ignore_spaces=True))
        self.assertFalse(StringUtils.is_palindrome("a b c", ignore_spaces=False))
    
    def test_string_utils_longest_palindrome(self):
        """Test longest palindromic substring"""
        # Basic cases
        self.assertEqual(StringUtils.longest_palindromic_substring("babad"), "bab")  # or "aba"
        self.assertEqual(StringUtils.longest_palindromic_substring("cbbd"), "bb")
        
        # Edge cases
        self.assertEqual(StringUtils.longest_palindromic_substring(""), "")
        self.assertEqual(StringUtils.longest_palindromic_substring("a"), "a")
        
        # Whole string is palindrome
        self.assertEqual(StringUtils.longest_palindromic_substring("racecar"), "racecar")
    
    def test_string_utils_anagrams(self):
        """Test anagram detection"""
        # Basic anagrams
        self.assertTrue(StringUtils.are_anagrams("listen", "silent"))
        self.assertTrue(StringUtils.are_anagrams("elbow", "below"))
        self.assertTrue(StringUtils.are_anagrams("A", "a"))
        
        # Non-anagrams
        self.assertFalse(StringUtils.are_anagrams("hello", "world"))
        self.assertFalse(StringUtils.are_anagrams("python", "java"))
        
        # Different lengths
        self.assertFalse(StringUtils.are_anagrams("ab", "abc"))
        
        # Case sensitivity
        self.assertFalse(StringUtils.are_anagrams("A", "a", ignore_case=False))
    
    def test_string_utils_find_anagrams(self):
        """Test finding anagrams in text"""
        text = "abab"
        pattern = "ab"
        
        anagrams = StringUtils.find_anagrams(text, pattern)
        # "ab" at index 0, "ba" at index 1, "ab" at index 2
        self.assertEqual(anagrams, [0, 1, 2])  
        
        # No anagrams
        text2 = "abcdef"
        pattern2 = "xyz"
        anagrams2 = StringUtils.find_anagrams(text2, pattern2)
        self.assertEqual(anagrams2, [])
        
        # Test with different patterns
        text3 = "abcabc"
        pattern3 = "bca"
        anagrams3 = StringUtils.find_anagrams(text3, pattern3)
        # "abc", "bca", "cab", "abc" are all anagrams of "bca"
        self.assertEqual(anagrams3, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
