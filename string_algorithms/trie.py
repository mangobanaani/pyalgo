class TrieNode:
    """Node in a Trie data structure"""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0  # Count of words ending at this node


class Trie:
    """
    Trie (Prefix Tree) implementation for efficient string storage and retrieval.
    
    Time Complexity:
    - Insert: O(m) where m is the length of the word
    - Search: O(m) where m is the length of the word
    - StartsWith: O(m) where m is the length of the prefix
    
    Space Complexity: O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
    """
    
    def __init__(self):
        """Initialize an empty Trie"""
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the Trie
        
        Args:
            word: String to insert into the Trie
        """
        if not word:
            return
        
        current = self.root
        for char in word.lower():
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        
        if not current.is_end_of_word:
            self.total_words += 1
        current.is_end_of_word = True
        current.word_count += 1
    
    def search(self, word: str) -> bool:
        """
        Search for a complete word in the Trie
        
        Args:
            word: String to search for
            
        Returns:
            True if word exists in Trie, False otherwise
        """
        if not word:
            return False
        
        current = self.root
        for char in word.lower():
            if char not in current.children:
                return False
            current = current.children[char]
        
        return current.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in the Trie starts with the given prefix
        
        Args:
            prefix: Prefix to search for
            
        Returns:
            True if any word starts with prefix, False otherwise
        """
        if not prefix:
            return True
        
        current = self.root
        for char in prefix.lower():
            if char not in current.children:
                return False
            current = current.children[char]
        
        return True
    
    def get_words_with_prefix(self, prefix: str) -> list:
        """
        Get all words in the Trie that start with the given prefix
        
        Args:
            prefix: Prefix to search for
            
        Returns:
            List of words starting with the prefix
        """
        words = []
        
        # Navigate to the prefix node
        current = self.root
        for char in prefix.lower():
            if char not in current.children:
                return words
            current = current.children[char]
        
        # Collect all words from this node
        self._collect_words(current, prefix.lower(), words)
        return words
    
    def _collect_words(self, node: TrieNode, current_word: str, words: list) -> None:
        """
        Helper method to collect all words from a given node
        
        Args:
            node: Current TrieNode
            current_word: Word built so far
            words: List to collect words in
        """
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, words)
    
    def delete(self, word: str) -> bool:
        """
        Delete a word from the Trie
        
        Args:
            word: Word to delete
            
        Returns:
            True if word was deleted, False if word wasn't found
        """
        if not word or not self.search(word):
            return False
        
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                # We've reached the end of the word
                if node.is_end_of_word:
                    node.is_end_of_word = False
                    node.word_count -= 1
                    self.total_words -= 1
                    # Return True if node has no children (can be deleted)
                    return len(node.children) == 0
                return False
            
            char = word[index].lower()
            child_node = node.children.get(char)
            
            if not child_node:
                return False
            
            should_delete_child = _delete_helper(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                # Return True if current node should be deleted
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        _delete_helper(self.root, word, 0)
        return True
    
    def count_words(self) -> int:
        """Return the total number of words in the Trie"""
        return self.total_words
    
    def is_empty(self) -> bool:
        """Check if the Trie is empty"""
        return self.total_words == 0
    
    def get_all_words(self) -> list:
        """Get all words stored in the Trie"""
        return self.get_words_with_prefix("")
