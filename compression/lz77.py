"""
LZ77 Compression Algorithm

A dictionary-based compression algorithm that encodes repeated occurrences 
of data as references to previous occurrences in the data.
"""

def lz77_encode(data, window_size=4096, lookahead_buffer_size=16):
    """
    Compress data using the LZ77 algorithm.
    
    :param data: String to compress
    :param window_size: Size of the sliding window
    :param lookahead_buffer_size: Size of the lookahead buffer
    :return: List of (offset, length, next_char) tuples
    """
    result = []
    position = 0
    
    while position < len(data):
        # Find the longest match in the window
        match = find_longest_match(data, position, window_size, lookahead_buffer_size)
        
        if match[1] > 0:  # If we found a match
            # (offset, length, next_char)
            result.append((match[0], match[1], data[position + match[1]] if position + match[1] < len(data) else ''))
            position += match[1] + 1
        else:
            # No match found, output a literal
            result.append((0, 0, data[position]))
            position += 1
    
    return result

def find_longest_match(data, current_position, window_size, lookahead_buffer_size):
    """
    Find the longest match in the sliding window.
    
    :param data: Input data
    :param current_position: Current position in the data
    :param window_size: Size of the sliding window
    :param lookahead_buffer_size: Size of the lookahead buffer
    :return: Tuple of (offset, length)
    """
    end_of_buffer = min(current_position + lookahead_buffer_size, len(data))
    
    # If we're at the end of the data, return no match
    if current_position >= len(data):
        return (0, 0)
    
    best_match_distance = 0
    best_match_length = 0
    
    # For each position in the sliding window
    start_of_window = max(0, current_position - window_size)
    
    # Try to find a match for the current lookahead buffer in the window
    for j in range(start_of_window, current_position):
        match_length = 0
        while (current_position + match_length < end_of_buffer and 
               data[j + match_length] == data[current_position + match_length]):
            match_length += 1
        
        # If this match is longer than any previous match
        if match_length > best_match_length:
            best_match_distance = current_position - j
            best_match_length = match_length
    
    return (best_match_distance, best_match_length)

def lz77_decode(encoded_data):
    """
    Decompress data that was compressed using the LZ77 algorithm.
    
    :param encoded_data: List of (offset, length, next_char) tuples
    :return: Original string
    """
    result = []
    
    for offset, length, next_char in encoded_data:
        # If length is 0, just add the next character
        if length == 0:
            result.append(next_char)
        else:
            # Copy 'length' characters from 'offset' positions back
            start = len(result) - offset
            for i in range(length):
                result.append(result[start + i])
            
            # Add the next character if it exists
            if next_char:
                result.append(next_char)
    
    return ''.join(result)
