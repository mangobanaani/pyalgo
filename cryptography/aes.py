"""
AES (Advanced Encryption Standard)

A symmetric encryption algorithm widely used to secure sensitive data.
This is a simplified implementation for educational purposes and should NOT
be used for actual encryption needs.

For real applications, use a secure, tested library like cryptography, pycrypto, or pycryptodome.
"""

# S-Box for SubBytes step (simplified version)
S_BOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
]

# Inverse S-Box for InvSubBytes
INV_S_BOX = [
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
]

# Rcon array for key expansion
RCON = [
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
]

def sub_bytes(state):
    """
    Apply the S-box to each byte of the state.
    
    :param state: 4x4 byte array representing the state
    :return: State after SubBytes transformation
    """
    for i in range(4):
        for j in range(4):
            state[i][j] = S_BOX[state[i][j]]
    return state

def inv_sub_bytes(state):
    """
    Apply the inverse S-box to each byte of the state.
    
    :param state: 4x4 byte array representing the state
    :return: State after InvSubBytes transformation
    """
    for i in range(4):
        for j in range(4):
            state[i][j] = INV_S_BOX[state[i][j]]
    return state

def shift_rows(state):
    """
    Cyclically shift the rows of the state array.
    
    :param state: 4x4 byte array representing the state
    :return: State after ShiftRows transformation
    """
    # Row 0: No shift
    # Row 1: Shift left by 1
    state[1] = state[1][1:] + state[1][:1]
    # Row 2: Shift left by 2
    state[2] = state[2][2:] + state[2][:2]
    # Row 3: Shift left by 3
    state[3] = state[3][3:] + state[3][:3]
    return state

def inv_shift_rows(state):
    """
    Apply the inverse ShiftRows transformation.
    
    :param state: 4x4 byte array representing the state
    :return: State after InvShiftRows transformation
    """
    # Row 0: No shift
    # Row 1: Shift right by 1
    state[1] = state[1][3:] + state[1][:3]
    # Row 2: Shift right by 2
    state[2] = state[2][2:] + state[2][:2]
    # Row 3: Shift right by 3
    state[3] = state[3][1:] + state[3][:1]
    return state

def galois_multiply(a, b):
    """
    Multiply two bytes in the Galois Field GF(2^8).
    
    :param a: First byte
    :param b: Second byte
    :return: Product in GF(2^8)
    """
    p = 0
    for i in range(8):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set:
            a ^= 0x1B  # Irreducible polynomial x^8 + x^4 + x^3 + x + 1
        b >>= 1
    return p % 256

def mix_columns(state):
    """
    Apply the MixColumns transformation.
    
    :param state: 4x4 byte array representing the state
    :return: State after MixColumns transformation
    """
    for i in range(4):
        col = [state[j][i] for j in range(4)]
        state[0][i] = galois_multiply(2, col[0]) ^ galois_multiply(3, col[1]) ^ col[2] ^ col[3]
        state[1][i] = col[0] ^ galois_multiply(2, col[1]) ^ galois_multiply(3, col[2]) ^ col[3]
        state[2][i] = col[0] ^ col[1] ^ galois_multiply(2, col[2]) ^ galois_multiply(3, col[3])
        state[3][i] = galois_multiply(3, col[0]) ^ col[1] ^ col[2] ^ galois_multiply(2, col[3])
    return state

def inv_mix_columns(state):
    """
    Apply the inverse MixColumns transformation.
    
    :param state: 4x4 byte array representing the state
    :return: State after InvMixColumns transformation
    """
    for i in range(4):
        col = [state[j][i] for j in range(4)]
        state[0][i] = galois_multiply(14, col[0]) ^ galois_multiply(11, col[1]) ^ galois_multiply(13, col[2]) ^ galois_multiply(9, col[3])
        state[1][i] = galois_multiply(9, col[0]) ^ galois_multiply(14, col[1]) ^ galois_multiply(11, col[2]) ^ galois_multiply(13, col[3])
        state[2][i] = galois_multiply(13, col[0]) ^ galois_multiply(9, col[1]) ^ galois_multiply(14, col[2]) ^ galois_multiply(11, col[3])
        state[3][i] = galois_multiply(11, col[0]) ^ galois_multiply(13, col[1]) ^ galois_multiply(9, col[2]) ^ galois_multiply(14, col[3])
    return state

def add_round_key(state, round_key):
    """
    XOR the state array with a round key.
    
    :param state: 4x4 byte array representing the state
    :param round_key: 4x4 byte array representing a round key
    :return: State after AddRoundKey transformation
    """
    for i in range(4):
        for j in range(4):
            state[i][j] ^= round_key[i][j]
    return state

def expand_key(key, rounds):
    """
    Expand the initial key into a key schedule.
    
    :param key: 16-byte array representing the initial key
    :param rounds: Number of rounds (10 for AES-128)
    :return: Expanded key schedule
    """
    # Convert key to 4x4 matrix (column-major order)
    key_matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            key_matrix[j][i] = key[i * 4 + j]
    
    # Expand key
    expanded_key = [key_matrix]
    
    for i in range(rounds):
        last_key = expanded_key[-1]
        new_key = [[0 for _ in range(4)] for _ in range(4)]
        
        # Calculate the first column of the new key
        temp = [last_key[j][3] for j in range(4)]  # Last column of previous key
        
        # Rotate left
        temp = temp[1:] + temp[:1]
        
        # Apply S-Box
        for j in range(4):
            temp[j] = S_BOX[temp[j]]
        
        # XOR with first column of previous key and Rcon
        for j in range(4):
            if j == 0:
                new_key[j][0] = last_key[j][0] ^ temp[j] ^ RCON[i]
            else:
                new_key[j][0] = last_key[j][0] ^ temp[j]
        
        # Calculate remaining columns
        for j in range(1, 4):
            for k in range(4):
                new_key[k][j] = new_key[k][j-1] ^ last_key[k][j]
        
        expanded_key.append(new_key)
    
    return expanded_key

def text_to_matrix(text):
    """
    Convert a 16-byte text to a 4x4 matrix.
    
    :param text: 16-byte array
    :return: 4x4 matrix (column-major)
    """
    matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            matrix[j][i] = text[i * 4 + j]
    return matrix

def matrix_to_text(matrix):
    """
    Convert a 4x4 matrix to a 16-byte array.
    
    :param matrix: 4x4 matrix (column-major)
    :return: 16-byte array
    """
    text = bytearray(16)
    for i in range(4):
        for j in range(4):
            text[i * 4 + j] = matrix[j][i]
    return text

def aes_encrypt_block(plaintext_block, key):
    """
    Encrypt a single 16-byte block using AES.
    
    :param plaintext_block: 16-byte block of plaintext
    :param key: 16-byte encryption key
    :return: 16-byte block of ciphertext
    """
    # Number of rounds for AES-128
    rounds = 10
    
    # Convert the plaintext block to a state array
    state = text_to_matrix(plaintext_block)
    
    # Generate the key schedule
    key_schedule = expand_key(key, rounds)
    
    # Initial round key addition
    state = add_round_key(state, key_schedule[0])
    
    # Main rounds
    for i in range(1, rounds):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, key_schedule[i])
    
    # Final round (no mix_columns)
    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, key_schedule[rounds])
    
    # Convert the state array back to a block
    return matrix_to_text(state)

def aes_decrypt_block(ciphertext_block, key):
    """
    Decrypt a single 16-byte block using AES.
    
    :param ciphertext_block: 16-byte block of ciphertext
    :param key: 16-byte encryption key
    :return: 16-byte block of plaintext
    """
    # Number of rounds for AES-128
    rounds = 10
    
    # Convert the ciphertext block to a state array
    state = text_to_matrix(ciphertext_block)
    
    # Generate the key schedule
    key_schedule = expand_key(key, rounds)
    
    # Initial round key addition
    state = add_round_key(state, key_schedule[rounds])
    
    # Main rounds (in reverse)
    for i in range(rounds-1, 0, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, key_schedule[i])
        state = inv_mix_columns(state)
    
    # Final round
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, key_schedule[0])
    
    # Convert the state array back to a block
    return matrix_to_text(state)

def pkcs7_pad(data, block_size=16):
    """
    Pad the data using PKCS#7.
    
    :param data: Data to pad
    :param block_size: Block size (16 bytes for AES)
    :return: Padded data
    """
    pad_len = block_size - (len(data) % block_size)
    padding = bytes([pad_len]) * pad_len
    return data + padding

def pkcs7_unpad(data):
    """
    Remove PKCS#7 padding.
    
    :param data: Padded data
    :return: Original data without padding
    """
    pad_len = data[-1]
    return data[:-pad_len]

def aes_encrypt_cbc(plaintext, key, iv):
    """
    Encrypt data using AES in CBC mode.
    
    :param plaintext: Data to encrypt
    :param key: 16-byte encryption key
    :param iv: 16-byte initialization vector
    :return: Encrypted data
    """
    # Pad the plaintext
    padded = pkcs7_pad(plaintext)
    
    # Split into blocks
    blocks = [padded[i:i+16] for i in range(0, len(padded), 16)]
    
    # Encrypt each block
    ciphertext = bytearray()
    previous_block = iv
    
    for block in blocks:
        # XOR with the previous block or IV
        xored = bytes(a ^ b for a, b in zip(block, previous_block))
        
        # Encrypt the block
        encrypted_block = aes_encrypt_block(xored, key)
        ciphertext.extend(encrypted_block)
        
        # Update previous block
        previous_block = encrypted_block
    
    return bytes(ciphertext)

def aes_decrypt_cbc(ciphertext, key, iv):
    """
    Decrypt data using AES in CBC mode.
    
    :param ciphertext: Data to decrypt
    :param key: 16-byte encryption key
    :param iv: 16-byte initialization vector
    :return: Decrypted data
    """
    # Check that ciphertext length is a multiple of block size
    if len(ciphertext) % 16 != 0:
        raise ValueError("Ciphertext length must be a multiple of 16 bytes")
    
    # Split into blocks
    blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
    
    # Decrypt each block
    plaintext = bytearray()
    previous_block = iv
    
    for block in blocks:
        # Decrypt the block
        decrypted = aes_decrypt_block(block, key)
        
        # XOR with the previous block or IV
        xored = bytes(a ^ b for a, b in zip(decrypted, previous_block))
        plaintext.extend(xored)
        
        # Update previous block
        previous_block = block
    
    # Remove padding
    return pkcs7_unpad(plaintext)

def generate_key():
    """
    Generate a random 128-bit (16-byte) key.
    
    :return: Random key
    """
    import os
    return os.urandom(16)

def generate_iv():
    """
    Generate a random 128-bit (16-byte) initialization vector.
    
    :return: Random IV
    """
    import os
    return os.urandom(16)
