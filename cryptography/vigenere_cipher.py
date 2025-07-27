"""
Vigenère Cipher

A polyalphabetic substitution cipher that uses a series of Caesar ciphers based on 
the letters of a keyword. It was considered unbreakable for three centuries.
"""

def vigenere_encrypt(plaintext, key):
    """
    Encrypt the given text using Vigenère cipher.
    
    :param plaintext: Text to be encrypted
    :param key: Keyword for encryption
    :return: Encrypted text
    """
    # Convert key to uppercase and remove non-alphabetic characters
    key = ''.join(c.upper() for c in key if c.isalpha())
    if not key:
        raise ValueError("Key must contain at least one alphabetic character")
    
    ciphertext = ""
    key_index = 0
    
    for char in plaintext:
        if char.isalpha():
            # Get the shift from the current key character
            # A=0, B=1, ..., Z=25
            shift = ord(key[key_index % len(key)]) - ord('A')
            
            # Determine if character is uppercase or lowercase
            ascii_offset = ord('A') if char.isupper() else ord('a')
            
            # Apply the shift
            shifted = (ord(char) - ascii_offset + shift) % 26 + ascii_offset
            
            ciphertext += chr(shifted)
            
            # Move to the next key character
            key_index += 1
        else:
            # If not a letter, keep it as is
            ciphertext += char
    
    return ciphertext

def vigenere_decrypt(ciphertext, key):
    """
    Decrypt the given text that was encrypted with Vigenère cipher.
    
    :param ciphertext: Text to be decrypted
    :param key: Keyword used for encryption
    :return: Decrypted text
    """
    # Convert key to uppercase and remove non-alphabetic characters
    key = ''.join(c.upper() for c in key if c.isalpha())
    if not key:
        raise ValueError("Key must contain at least one alphabetic character")
    
    plaintext = ""
    key_index = 0
    
    for char in ciphertext:
        if char.isalpha():
            # Get the shift from the current key character
            # A=0, B=1, ..., Z=25
            shift = ord(key[key_index % len(key)]) - ord('A')
            
            # Determine if character is uppercase or lowercase
            ascii_offset = ord('A') if char.isupper() else ord('a')
            
            # Apply the negative shift (for decryption)
            shifted = (ord(char) - ascii_offset - shift) % 26 + ascii_offset
            
            plaintext += chr(shifted)
            
            # Move to the next key character
            key_index += 1
        else:
            # If not a letter, keep it as is
            plaintext += char
    
    return plaintext

def find_repeated_sequences(text, min_length=3, max_length=5):
    """
    Find repeated sequences in the text which can help in determining the key length.
    
    :param text: Input text to analyze
    :param min_length: Minimum sequence length to consider
    :param max_length: Maximum sequence length to consider
    :return: Dictionary mapping sequences to lists of their positions
    """
    sequences = {}
    
    # Only consider alphabetic characters
    text = ''.join(c for c in text if c.isalpha()).upper()
    
    # Try all sequence lengths from min to max
    for length in range(min_length, max_length + 1):
        for i in range(len(text) - length + 1):
            seq = text[i:i+length]
            if seq in sequences:
                sequences[seq].append(i)
            else:
                sequences[seq] = [i]
    
    # Filter out sequences that don't repeat
    return {seq: positions for seq, positions in sequences.items() if len(positions) > 1}

def calculate_key_length(repeated_sequences):
    """
    Calculate possible key lengths based on the distances between repeated sequences.
    
    :param repeated_sequences: Dictionary mapping sequences to positions
    :return: List of possible key lengths sorted by likelihood
    """
    # Calculate distances between all occurrences of each sequence
    distances = []
    for positions in repeated_sequences.values():
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = positions[j] - positions[i]
                if distance > 1:  # Ignore adjacent repeats
                    distances.append(distance)
    
    # Find factors of the distances
    all_factors = []
    for distance in distances:
        factors = []
        for i in range(1, distance + 1):
            if distance % i == 0:
                factors.append(i)
        all_factors.extend(factors)
    
    # Count occurrences of each factor
    factor_counts = {}
    for factor in all_factors:
        if factor > 1:  # Ignore factor 1
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    # Sort factors by frequency, from most common to least common
    sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
    return [factor for factor, count in sorted_factors]

def vigenere_crack(ciphertext, max_key_length=10):
    """
    Attempt to crack the Vigenère cipher using Kasiski examination.
    
    :param ciphertext: Encrypted text to crack
    :param max_key_length: Maximum key length to consider
    :return: Tuple of (most likely plaintext, most likely key)
    """
    # Remove non-alphabetic characters
    cleaned_text = ''.join(c for c in ciphertext if c.isalpha()).upper()
    
    # Find repeated sequences
    repeated = find_repeated_sequences(cleaned_text)
    
    # Calculate possible key lengths
    possible_lengths = calculate_key_length(repeated)
    
    # If no key length found, try all up to max_key_length
    if not possible_lengths:
        possible_lengths = list(range(1, max_key_length + 1))
    
    # Trim to max_key_length
    possible_lengths = [l for l in possible_lengths if l <= max_key_length]
    
    # TODO: Implement full frequency analysis for each key length
    # This would analyze each "column" of ciphertext when grouped by key length
    # For this simplified implementation, we'll return a placeholder
    
    return "Cracking Vigenère cipher requires frequency analysis for each key length", possible_lengths
