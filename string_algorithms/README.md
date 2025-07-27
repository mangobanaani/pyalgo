# String Algorithms

Comprehensive collection of string processing algorithms for pattern matching, text analysis, and advanced string structures.

## Algorithms Included

### Pattern Matching Algorithms

#### KMP (Knuth-Morris-Pratt) - `kmp.py`
- **Time Complexity**: O(n + m) where n is text length, m is pattern length
- **Space Complexity**: O(m) for failure function
- **Description**: Uses preprocessed failure function to avoid redundant comparisons
- **Advantage**: Optimal linear time, no backtracking in text
- **Use Case**: Single pattern search in large texts

#### Boyer-Moore - `boyer_moore.py`
- **Time Complexity**: O(nm) worst case, O(n/m) best case
- **Space Complexity**: O(σ) where σ is alphabet size
- **Description**: Scans pattern from right to left, uses bad character and good suffix heuristics
- **Advantage**: Sublinear time in practice, excellent for large alphabets
- **Use Case**: Text editors, grep utilities

#### Rabin-Karp - `rabin_karp.py`
- **Time Complexity**: O(n + m) average, O(nm) worst case
- **Space Complexity**: O(1)
- **Description**: Uses rolling hash to compare pattern and text substrings
- **Advantage**: Easy to extend for multiple pattern search
- **Use Case**: Plagiarism detection, finding duplicate code

#### Z-Algorithm - `z_algorithm.py`
- **Time Complexity**: O(n) where n is string length
- **Space Complexity**: O(n) for Z-array
- **Description**: Computes Z-array containing longest substring starting from each position that matches prefix
- **Advantage**: Linear time, useful for various string problems
- **Use Case**: Pattern matching, string analysis

#### Aho-Corasick - `aho_corasick.py`
- **Time Complexity**: O(n + m + z) where z is number of pattern occurrences
- **Space Complexity**: O(ALPHABET_SIZE × number of states)
- **Description**: Efficiently finds all occurrences of multiple patterns simultaneously
- **Advantage**: Finds all patterns in single pass
- **Use Case**: Intrusion detection, bioinformatics, text filtering

### String Analysis Algorithms

#### Anagram Detection - `anagram_detection.py`
- **Algorithms**: Sorting-based, frequency counting, prime factorization
- **Time Complexity**: O(n log n) for sorting, O(n) for counting
- **Description**: Various methods to detect if two strings are anagrams
- **Use Case**: Word games, linguistic analysis

#### Palindrome Checking - `palindrome_checking.py`
- **Algorithms**: Two-pointer, recursive, iterative approaches
- **Time Complexity**: O(n)
- **Description**: Efficient methods to check if string reads same forwards and backwards
- **Use Case**: DNA sequence analysis, word puzzles

#### Longest Common Substring - `longest_common_substring.py`
- **Time Complexity**: O(nm) using dynamic programming
- **Space Complexity**: O(nm) or O(min(n,m)) with space optimization
- **Description**: Finds longest contiguous substring common to two strings
- **Use Case**: DNA sequence alignment, file comparison

#### Longest Palindromic Substring - `longest_palindromic_substring.py`
- **Algorithms**: Expand around centers, Manacher's algorithm
- **Time Complexity**: O(n²) for expansion, O(n) for Manacher's
- **Description**: Finds longest palindromic substring in given string
- **Use Case**: Bioinformatics, pattern recognition

### String Compression

#### String Compression - `string_compression.py`
- **Algorithm**: Run-length encoding (RLE)
- **Time Complexity**: O(n)
- **Description**: Compresses strings by encoding consecutive repeated characters
- **Use Case**: Simple data compression, image processing

### Advanced String Structures

#### Trie (Prefix Tree) - `trie.py`
- **Time Complexity**: O(m) for insert/search where m is key length
- **Space Complexity**: O(ALPHABET_SIZE × N × M) worst case
- **Description**: Tree structure for efficient string storage and prefix operations
- **Operations**: Insert, search, prefix search, delete
- **Use Case**: Autocomplete, spell checkers, IP routing

#### Suffix Array - `suffix_array.py`
- **Time Complexity**: O(n log n) for construction, O(log n) for search
- **Space Complexity**: O(n)
- **Description**: Sorted array of all suffixes, space-efficient alternative to suffix tree
- **Use Case**: Substring search, longest common substring, bioinformatics

## Usage Examples

### Pattern Matching
```python
from string_algorithms.kmp import kmp_search
from string_algorithms.boyer_moore import boyer_moore_search
from string_algorithms.aho_corasick import AhoCorasick

# KMP Pattern Search
text = "ABABDABACDABABCABCABCABCABC"
pattern = "ABABCAB"
positions = kmp_search(text, pattern)
print(f"KMP found pattern at positions: {positions}")

# Boyer-Moore Search
positions = boyer_moore_search(text, pattern)
print(f"Boyer-Moore found pattern at positions: {positions}")

# Multiple Pattern Search with Aho-Corasick
patterns = ["ABABCAB", "ABC", "CAB"]
ac = AhoCorasick()
for pattern in patterns:
    ac.add_pattern(pattern)
ac.build_failure_links()

matches = ac.search(text)
print(f"Aho-Corasick found matches: {matches}")
```

### String Analysis
```python
from string_algorithms.anagram_detection import are_anagrams
from string_algorithms.palindrome_checking import is_palindrome
from string_algorithms.longest_common_substring import lcs_length

# Anagram Detection
word1, word2 = "listen", "silent"
print(f"'{word1}' and '{word2}' are anagrams: {are_anagrams(word1, word2)}")

# Palindrome Checking
text = "racecar"
print(f"'{text}' is palindrome: {is_palindrome(text)}")

# Longest Common Substring
str1, str2 = "ABABC", "BABCA"
length = lcs_length(str1, str2)
print(f"Longest common substring length: {length}")
```

### Advanced Structures
```python
from string_algorithms.trie import Trie
from string_algorithms.suffix_array import SuffixArray

# Trie Operations
trie = Trie()
words = ["cat", "car", "card", "care", "careful"]
for word in words:
    trie.insert(word)

print(f"'car' in trie: {trie.search('car')}")
print(f"Words with prefix 'car': {trie.words_with_prefix('car')}")

# Suffix Array
text = "banana"
sa = SuffixArray(text)
print(f"Suffix array for '{text}': {sa.suffix_array}")
print(f"Searching 'ana': {sa.search('ana')}")
```

### String Compression
```python
from string_algorithms.string_compression import run_length_encode, run_length_decode

# Run-Length Encoding
original = "AAABBBCCDAA"
compressed = run_length_encode(original)
print(f"Original: {original}")
print(f"Compressed: {compressed}")

decompressed = run_length_decode(compressed)
print(f"Decompressed: {decompressed}")
```

## Algorithm Performance Comparison

### Pattern Matching Algorithms

| Algorithm | Preprocessing | Search | Best Case | Worst Case | Space |
|-----------|---------------|--------|-----------|------------|-------|
| Naive | O(1) | O(nm) | O(n) | O(nm) | O(1) |
| KMP | O(m) | O(n) | O(n) | O(n+m) | O(m) |
| Boyer-Moore | O(m + σ) | O(n/m) | O(n/m) | O(nm) | O(σ) |
| Rabin-Karp | O(m) | O(n) | O(n+m) | O(nm) | O(1) |
| Aho-Corasick | O(Σm) | O(n) | O(n+z) | O(n+z) | O(σ×states) |

*Where n = text length, m = pattern length, σ = alphabet size, z = number of matches*

### String Structure Operations

| Structure | Insert | Search | Delete | Space | Best Use Case |
|-----------|--------|--------|--------|-------|---------------|
| Trie | O(m) | O(m) | O(m) | O(σ×N×M) | Prefix operations |
| Suffix Array | O(n log n) | O(log n) | N/A | O(n) | Substring queries |
| Hash Table | O(1) avg | O(1) avg | O(1) avg | O(n) | Exact matching |

## Applications by Domain

### Text Processing
- **Search Engines**: Boyer-Moore for fast text search
- **Text Editors**: KMP for find/replace operations
- **Log Analysis**: Aho-Corasick for multiple keyword detection

### Bioinformatics
- **DNA Sequencing**: Suffix arrays for genome analysis
- **Protein Matching**: Z-algorithm for sequence alignment
- **Gene Finding**: Pattern matching algorithms

### Security
- **Intrusion Detection**: Aho-Corasick for signature matching
- **Virus Scanning**: Multiple pattern matching
- **Data Loss Prevention**: String analysis for sensitive data

### Web Development
- **URL Routing**: Trie for efficient path matching
- **Autocomplete**: Trie for suggestion systems
- **Content Filtering**: Pattern matching for content moderation

## Algorithm Selection Guide

### Choose KMP When:
- Single pattern search in large texts
- Preprocessing time is acceptable
- Guaranteed linear time is required

### Choose Boyer-Moore When:
- Large alphabet (English text)
- Pattern is relatively long
- Sublinear performance is desired

### Choose Rabin-Karp When:
- Multiple pattern search
- Simple implementation is preferred
- Hash-based approaches are suitable

### Choose Aho-Corasick When:
- Multiple patterns must be found simultaneously
- Building automaton once for many searches
- All occurrences of all patterns are needed

### Choose Trie When:
- Prefix-based operations are common
- Autocomplete functionality is needed
- Set of strings changes dynamically

### Choose Suffix Array When:
- Substring queries are frequent
- Space efficiency is important
- Text is relatively static

## Advanced Topics

### Optimization Techniques
- **Preprocessing**: Reduce online computation time
- **Space-Time Tradeoffs**: Suffix arrays vs suffix trees
- **Cache Optimization**: Locality-aware implementations

### Parallel Algorithms
- **Parallel KMP**: Multiple threads for large texts
- **Distributed Pattern Matching**: Map-reduce approaches
- **GPU Acceleration**: SIMD operations for string processing

### Approximate Matching
- **Edit Distance**: Fuzzy string matching
- **k-Mismatch**: Allow up to k character differences
- **Regular Expressions**: Pattern matching with wildcards
