"""
Huffman Coding Compression

A lossless data compression algorithm that uses variable-length codes to represent 
symbols. More frequent symbols are assigned shorter codes, less frequent symbols 
are assigned longer codes.
"""
import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    """Node for the Huffman tree."""
    def __init__(self, symbol=None, frequency=0):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        # Define comparison for priority queue
        return self.frequency < other.frequency

def build_frequency_table(data):
    """
    Build a frequency table for the input data.
    
    :param data: Input string or bytes
    :return: Dictionary mapping symbols to their frequencies
    """
    return Counter(data)

def build_huffman_tree(frequency_table):
    """
    Build a Huffman tree from a frequency table.
    
    :param frequency_table: Dictionary mapping symbols to frequencies
    :return: Root node of the Huffman tree
    """
    # Create a leaf node for each symbol
    priority_queue = [HuffmanNode(symbol, freq) for symbol, freq in frequency_table.items()]
    heapq.heapify(priority_queue)
    
    # Build the tree by combining the two lowest frequency nodes
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        # Create a new internal node with these two nodes as children
        # and with frequency equal to the sum of their frequencies
        parent = HuffmanNode(frequency=left.frequency + right.frequency)
        parent.left = left
        parent.right = right
        
        heapq.heappush(priority_queue, parent)
    
    # Return the root of the Huffman tree
    return priority_queue[0] if priority_queue else None

def build_encoding_table(root):
    """
    Build a table mapping symbols to their Huffman codes.
    
    :param root: Root node of the Huffman tree
    :return: Dictionary mapping symbols to their codes
    """
    codes = {}
    
    def traverse_tree(node, code=""):
        if node:
            # If this is a leaf node, store the code
            if node.symbol is not None:
                codes[node.symbol] = code
            
            # Traverse left (add '0')
            traverse_tree(node.left, code + "0")
            
            # Traverse right (add '1')
            traverse_tree(node.right, code + "1")
    
    traverse_tree(root)
    return codes

def huffman_encode(data):
    """
    Compress data using Huffman coding.
    
    :param data: Input string or bytes to compress
    :return: Tuple of (encoded data, encoding table)
    """
    if not data:
        return "", {}
    
    # Build frequency table and Huffman tree
    frequency_table = build_frequency_table(data)
    tree = build_huffman_tree(frequency_table)
    
    # Build encoding table
    encoding_table = build_encoding_table(tree)
    
    # Encode the data
    encoded_data = ''.join(encoding_table[symbol] for symbol in data)
    
    return encoded_data, encoding_table

def huffman_decode(encoded_data, encoding_table):
    """
    Decompress data that was compressed using Huffman coding.
    
    :param encoded_data: Encoded binary string
    :param encoding_table: Dictionary mapping symbols to their codes
    :return: Original data
    """
    if not encoded_data:
        return ""
    
    # Create a reverse lookup table
    decoding_table = {code: symbol for symbol, code in encoding_table.items()}
    
    result = []
    current_code = ""
    
    for bit in encoded_data:
        current_code += bit
        
        # Check if the current code matches any symbol
        if current_code in decoding_table:
            result.append(decoding_table[current_code])
            current_code = ""
    
    return ''.join(result)
