"""
Caesar Cipher

A substitution cipher where each letter in the plaintext is replaced by a letter 
some fixed number of positions down the alphabet.
"""

def caesar_encrypt(plaintext, shift):
    """
    Encrypt the given text using Caesar cipher.
    
    :param plaintext: Text to be encrypted
    :param shift: The number of positions to shift each letter
    :return: Encrypted text
    """
    ciphertext = ""
    
    for char in plaintext:
        if char.isalpha():
            # Determine if character is uppercase or lowercase
            ascii_offset = ord('A') if char.isupper() else ord('a')
            
            # Apply the shift
            # The modulo 26 ensures the shift wraps around the alphabet
            shifted = (ord(char) - ascii_offset + shift) % 26 + ascii_offset
            
            ciphertext += chr(shifted)
        else:
            # If not a letter, keep it as is
            ciphertext += char
    
    return ciphertext

def caesar_decrypt(ciphertext, shift):
    """
    Decrypt the given text that was encrypted with Caesar cipher.
    
    :param ciphertext: Text to be decrypted
    :param shift: The number of positions the letters were shifted
    :return: Decrypted text
    """
    # Decryption is just encryption with the negative shift
    return caesar_encrypt(ciphertext, -shift)

def caesar_crack(ciphertext, language='english'):
    """
    Attempt to crack the Caesar cipher by trying all possible shifts
    and returning the most likely plaintext.
    
    :param ciphertext: Encrypted text to crack
    :param language: Language of the expected plaintext (for letter frequency analysis)
    :return: Tuple of (most likely plaintext, shift used)
    """
    # Common letter frequencies in English
    english_letter_freq = {
        'e': 12.02, 't': 9.10, 'a': 8.12, 'o': 7.68, 'i': 7.31, 'n': 6.95,
        's': 6.28, 'r': 6.02, 'h': 5.92, 'd': 4.32, 'l': 3.98, 'u': 2.88,
        'c': 2.71, 'm': 2.61, 'f': 2.30, 'y': 2.11, 'w': 2.09, 'g': 2.03,
        'p': 1.82, 'b': 1.49, 'v': 1.11, 'k': 0.69, 'x': 0.17, 'q': 0.11,
        'j': 0.10, 'z': 0.07
    }
    
    # Try all possible shifts and score them based on letter frequencies
    best_score = float('-inf')
    best_shift = 0
    best_plaintext = ""
    
    for shift in range(26):
        # Decrypt with the current shift
        plaintext = caesar_decrypt(ciphertext, shift)
        
        # Score the plaintext based on letter frequency
        score = 0
        for char in plaintext.lower():
            if char.isalpha():
                score += english_letter_freq.get(char, 0)
        
        # Update best result if this score is better
        if score > best_score:
            best_score = score
            best_shift = shift
            best_plaintext = plaintext
    
    return best_plaintext, best_shift
