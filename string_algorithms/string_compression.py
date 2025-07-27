def run_length_encoding(s: str) -> str:
    """
    Compresses a string using Run-Length Encoding.
    For example, 'AAABBBCCC' becomes 'A3B3C3'.
    
    :param s: Input string to compress
    :return: Compressed string
    """
    if not s:
        return ""
    
    result = []
    count = 1
    current_char = s[0]
    
    # Iterate through the string starting from the second character
    for i in range(1, len(s)):
        # If the current character is the same as the previous one, increment count
        if s[i] == current_char:
            count += 1
        else:
            # Append the character and its count to the result
            result.append(current_char + str(count))
            current_char = s[i]
            count = 1
    
    # Append the last character and its count
    result.append(current_char + str(count))
    
    compressed = ''.join(result)
    
    # Return the compressed string only if it's shorter than the original
    return compressed if len(compressed) < len(s) else s

def run_length_decoding(s: str) -> str:
    """
    Decompresses a string that was compressed using Run-Length Encoding.
    For example, 'A3B3C3' becomes 'AAABBBCCC'.
    
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
        
        # Find all consecutive digits after the character
        count_str = ""
        while i < len(s) and s[i].isdigit():
            count_str += s[i]
            i += 1
        
        # If there are no digits, count is 1 (implicit)
        count = int(count_str) if count_str else 1
        
        # Append the character repeated count times
        result.append(char * count)
    
    return ''.join(result)
