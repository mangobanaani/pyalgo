import heapq
from collections import Counter, namedtuple

class Node(namedtuple("Node", ["char", "freq", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(s):
    """
    Generate Huffman codes for characters in a string.

    Args:
        s (str): Input string.

    Returns:
        dict: A dictionary mapping characters to their Huffman codes.
    """
    if not s:
        return {}

    # Count frequency of each character
    freq = Counter(s)

    # Create a priority queue (min-heap)
    heap = [Node(char, freq, None, None) for char, freq in freq.items()]
    heapq.heapify(heap)

    # Build the Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)

    # Generate Huffman codes
    root = heap[0]
    codes = {}

    def generate_codes(node, code):
        if node.char is not None:
            codes[node.char] = code
            return
        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")

    generate_codes(root, "")
    return codes
