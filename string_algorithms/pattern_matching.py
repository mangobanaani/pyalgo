class KMPPatternMatcher:
    """
    Knuth-Morris-Pratt (KMP) algorithm for efficient pattern matching.
    
    Time Complexity: O(n + m) where n is text length, m is pattern length
    Space Complexity: O(m) for the failure function
    
    Advantage over naive approach: No backtracking in the text
    """
    
    @staticmethod
    def compute_lps_array(pattern: str) -> list:
        """
        Compute the Longest Proper Prefix which is also Suffix (LPS) array
        
        Args:
            pattern: Pattern string
            
        Returns:
            LPS array where lps[i] = length of longest proper prefix of pattern[0:i+1]
            which is also a suffix of pattern[0:i+1]
        """
        m = len(pattern)
        lps = [0] * m
        length = 0  # Length of previous longest prefix suffix
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    # Consider the previous longest prefix suffix
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    @staticmethod
    def search(text: str, pattern: str) -> list:
        """
        Search for all occurrences of pattern in text using KMP algorithm
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            
        Returns:
            List of starting indices where pattern is found
        """
        if not pattern or not text:
            return []
        
        n = len(text)
        m = len(pattern)
        
        # Compute LPS array
        lps = KMPPatternMatcher.compute_lps_array(pattern)
        
        occurrences = []
        i = 0  # Index for text
        j = 0  # Index for pattern
        
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                # Found a match
                occurrences.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                # Mismatch after j matches
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return occurrences
    
    @staticmethod
    def find_first(text: str, pattern: str) -> int:
        """
        Find the first occurrence of pattern in text
        
        Returns:
            Index of first occurrence, or -1 if not found
        """
        occurrences = KMPPatternMatcher.search(text, pattern)
        return occurrences[0] if occurrences else -1
    
    @staticmethod
    def count_occurrences(text: str, pattern: str) -> int:
        """
        Count the number of occurrences of pattern in text
        
        Returns:
            Number of occurrences
        """
        return len(KMPPatternMatcher.search(text, pattern))


class RollingHash:
    """
    Rolling hash implementation for efficient string hashing and comparison.
    Uses polynomial rolling hash with a prime base.
    """
    
    def __init__(self, base: int = 256, modulus: int = 10**9 + 7):
        """
        Initialize rolling hash parameters
        
        Args:
            base: Base for polynomial hash (typically 256 for ASCII)
            modulus: Prime modulus to prevent overflow
        """
        self.base = base
        self.modulus = modulus
    
    def compute_hash(self, s: str) -> int:
        """
        Compute hash value for a string
        
        Args:
            s: String to hash
            
        Returns:
            Hash value of the string
        """
        hash_value = 0
        for char in s:
            hash_value = (hash_value * self.base + ord(char)) % self.modulus
        return hash_value
    
    def rolling_search(self, text: str, pattern: str) -> list:
        """
        Search for pattern in text using rolling hash (Rabin-Karp algorithm)
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            
        Returns:
            List of starting indices where pattern is found
        """
        if not pattern or not text or len(pattern) > len(text):
            return []
        
        n = len(text)
        m = len(pattern)
        
        # Compute hash of pattern and first window of text
        pattern_hash = self.compute_hash(pattern)
        text_hash = self.compute_hash(text[:m])
        
        # Compute h = base^(m-1) % modulus for rolling
        h = 1
        for _ in range(m - 1):
            h = (h * self.base) % self.modulus
        
        occurrences = []
        
        # Check first window
        if text_hash == pattern_hash and text[:m] == pattern:
            occurrences.append(0)
        
        # Roll the hash over text
        for i in range(1, n - m + 1):
            # Remove leading character and add trailing character
            text_hash = (self.base * (text_hash - ord(text[i-1]) * h) + ord(text[i + m - 1])) % self.modulus
            
            # If hash values match, check characters
            if text_hash == pattern_hash and text[i:i+m] == pattern:
                occurrences.append(i)
        
        return occurrences


class StringUtils:
    """
    Collection of useful string processing utilities
    """
    
    @staticmethod
    def is_palindrome(s: str, ignore_case: bool = True, ignore_spaces: bool = True) -> bool:
        """
        Check if a string is a palindrome
        
        Args:
            s: String to check
            ignore_case: Whether to ignore case differences
            ignore_spaces: Whether to ignore spaces and punctuation
            
        Returns:
            True if string is a palindrome, False otherwise
        """
        if ignore_spaces:
            # Keep only alphanumeric characters
            cleaned = ''.join(char.lower() if ignore_case else char 
                            for char in s if char.isalnum())
            s = cleaned
        elif ignore_case:
            s = s.lower()
        
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    @staticmethod
    def longest_palindromic_substring(s: str) -> str:
        """
        Find the longest palindromic substring using expand around centers
        
        Args:
            s: Input string
            
        Returns:
            Longest palindromic substring
        """
        if not s:
            return ""
        
        start = 0
        max_len = 1
        
        def expand_around_center(left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1
        
        for i in range(len(s)):
            # Check for odd-length palindromes (center at i)
            len1 = expand_around_center(i, i)
            # Check for even-length palindromes (center between i and i+1)
            len2 = expand_around_center(i, i + 1)
            
            current_max = max(len1, len2)
            if current_max > max_len:
                max_len = current_max
                start = i - (current_max - 1) // 2
        
        return s[start:start + max_len]
    
    @staticmethod
    def are_anagrams(s1: str, s2: str, ignore_case: bool = True) -> bool:
        """
        Check if two strings are anagrams
        
        Args:
            s1, s2: Strings to compare
            ignore_case: Whether to ignore case
            
        Returns:
            True if strings are anagrams, False otherwise
        """
        if ignore_case:
            s1, s2 = s1.lower(), s2.lower()
        
        if len(s1) != len(s2):
            return False
        
        # Count character frequencies
        char_count = {}
        
        for char in s1:
            char_count[char] = char_count.get(char, 0) + 1
        
        for char in s2:
            if char not in char_count:
                return False
            char_count[char] -= 1
            if char_count[char] == 0:
                del char_count[char]
        
        return len(char_count) == 0
    
    @staticmethod
    def find_anagrams(text: str, pattern: str) -> list:
        """
        Find all anagrams of pattern in text using sliding window approach
        
        Args:
            text: Text to search in
            pattern: Pattern to find anagrams of
            
        Returns:
            List of starting indices where anagrams are found
        """
        if len(pattern) > len(text):
            return []
        
        result = []
        pattern_len = len(pattern)
        
        # Count frequency of characters in pattern
        pattern_count = {}
        for char in pattern:
            pattern_count[char] = pattern_count.get(char, 0) + 1
        
        # Sliding window approach
        window_count = {}
        
        # Initialize first window
        for i in range(pattern_len):
            char = text[i]
            window_count[char] = window_count.get(char, 0) + 1
        
        # Check first window
        if window_count == pattern_count:
            result.append(0)
        
        # Slide the window
        for i in range(pattern_len, len(text)):
            # Add new character to window
            new_char = text[i]
            window_count[new_char] = window_count.get(new_char, 0) + 1
            
            # Remove old character from window
            old_char = text[i - pattern_len]
            window_count[old_char] -= 1
            if window_count[old_char] == 0:
                del window_count[old_char]
            
            # Check if current window is an anagram
            if window_count == pattern_count:
                result.append(i - pattern_len + 1)
        
        return result
