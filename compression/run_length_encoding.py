"""
Run-Length Encoding (RLE) Compression

A simple lossless compression algorithm that encodes consecutive data elements 
(pixels, characters, etc.) by storing a single value and its count.
"""

def rle_encode(data):
    """
    Compress data using Run-Length Encoding.
    
    :param data: String or list of values to compress
    :return: Compressed data as a list of (value, count) tuples
    """
    if not data:
        return []
    
    result = []
    current_char = data[0]
    count = 1
    
    # Process each character in the data
    for i in range(1, len(data)):
        if data[i] == current_char:
            count += 1
        else:
            result.append((current_char, count))
            current_char = data[i]
            count = 1
    
    # Add the last run
    result.append((current_char, count))
    
    return result

def rle_decode(encoded_data):
    """
    Decompress data that was compressed using Run-Length Encoding.
    
    :param encoded_data: List of (value, count) tuples
    :return: Original data as a list or string (depending on input type)
    """
    if not encoded_data:
        return ""
    
    result = []
    
    # Process each (value, count) tuple
    for value, count in encoded_data:
        result.extend([value] * count)
    
    # Determine if the result should be a string or list
    if all(isinstance(item, str) and len(item) == 1 for item in result):
        return ''.join(result)
    else:
        return result

def rle_encode_string(s):
    """
    Compress a string using Run-Length Encoding and return a string.
    For example, "AAABBBCCC" becomes "A3B3C3".
    
    :param s: Input string to compress
    :return: Compressed string
    """
    if not s:
        return ""
    
    result = []
    current_char = s[0]
    count = 1
    
    # Process each character in the string
    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result.append(current_char + str(count))
            current_char = s[i]
            count = 1
    
    # Add the last run
    result.append(current_char + str(count))
    
    return ''.join(result)

def rle_decode_string(s):
    """
    Decompress a string that was compressed using Run-Length Encoding.
    For example, "A3B3C3" becomes "AAABBBCCC".
    
    :param s: Compressed string
    :return: Original decompressed string
    """
    if not s:
        return ""
    
    result = []
    i = 0
    
    while i < len(s):
        char = s[i]
        i += 1
        
        # Find all consecutive digits for the count
        count_str = ""
        while i < len(s) and s[i].isdigit():
            count_str += s[i]
            i += 1
        
        # Default to count 1 if no count found
        count = int(count_str) if count_str else 1
        
        # Append the character repeated count times
        result.append(char * count)
    
    return ''.join(result)
