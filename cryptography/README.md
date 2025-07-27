# Cryptography Algorithms

Implementation of classical and modern cryptographic algorithms for education and understanding of cryptographic principles.

## Algorithms Included

### Caesar Cipher (`caesar_cipher.py`)
- **Type**: Classical substitution cipher
- **Description**: Shifts each letter in plaintext by fixed number of positions in alphabet
- **Security**: Very weak, easily broken by frequency analysis
- **Key Space**: 25 possible keys (shift values 1-25)
- **Features**: Encryption, decryption, and cryptanalysis (cipher breaking)
- **Use Case**: Educational purposes, understanding basic cryptography concepts

### Vigenère Cipher (`vigenere_cipher.py`)
- **Type**: Polyalphabetic substitution cipher
- **Description**: Uses keyword to determine shift values, cycling through keyword letters
- **Security**: More secure than Caesar cipher but vulnerable to Kasiski examination
- **Key Space**: Depends on keyword length and alphabet size
- **Features**: Encryption, decryption with keyword-based shifting
- **Use Case**: Historical cryptography study, multi-character key systems

### RSA (`rsa.py`)
- **Type**: Public-key (asymmetric) cryptography
- **Description**: Uses mathematical properties of large prime factorization
- **Security**: Secure for sufficiently large key sizes (2048+ bits recommended)
- **Key Features**: Key pair generation, encryption, decryption, digital signatures
- **Components**: Public key (n, e), Private key (n, d)
- **Use Case**: Secure communication, digital signatures, key exchange

### AES (`aes.py`)
- **Type**: Symmetric block cipher
- **Description**: Advanced Encryption Standard, current U.S. government standard
- **Security**: Considered secure for all practical purposes
- **Key Sizes**: 128, 192, or 256 bits
- **Block Size**: 128 bits
- **Use Case**: High-speed encryption of large amounts of data

## Usage Examples

### Caesar Cipher
```python
from cryptography.caesar_cipher import caesar_encrypt, caesar_decrypt, caesar_crack

# Basic encryption and decryption
plaintext = "HELLO WORLD"
shift = 3
encrypted = caesar_encrypt(plaintext, shift)
print(f"Encrypted: {encrypted}")  # "KHOOR ZRUOG"

decrypted = caesar_decrypt(encrypted, shift)
print(f"Decrypted: {decrypted}")  # "HELLO WORLD"

# Cryptanalysis - breaking the cipher
ciphertext = "KHOOR ZRUOG"
cracked_text, found_shift = caesar_crack(ciphertext)
print(f"Cracked: '{cracked_text}' with shift {found_shift}")
```

### Vigenère Cipher
```python
from cryptography.vigenere_cipher import vigenere_encrypt, vigenere_decrypt

# Encryption with keyword
plaintext = "HELLO WORLD"
keyword = "KEY"
encrypted = vigenere_encrypt(plaintext, keyword)
print(f"Encrypted: {encrypted}")

# Decryption
decrypted = vigenere_decrypt(encrypted, keyword)
print(f"Decrypted: {decrypted}")

# Demonstrate polyalphabetic nature
print("Encryption process:")
for i, char in enumerate(plaintext.replace(" ", "")):
    key_char = keyword[i % len(keyword)]
    print(f"{char} + {key_char} = {encrypted[i]}")
```

### RSA Cryptography
```python
from cryptography.rsa import generate_keypair, rsa_encrypt, rsa_decrypt

# Generate RSA key pair
public_key, private_key = generate_keypair(keysize=1024)
print(f"Public key (n, e): {public_key}")
print(f"Private key (n, d): {private_key}")

# Encrypt message
message = 42  # RSA works on integers
encrypted = rsa_encrypt(message, public_key)
print(f"Encrypted: {encrypted}")

# Decrypt message
decrypted = rsa_decrypt(encrypted, private_key)
print(f"Decrypted: {decrypted}")

# Demonstrate public-key property
# Anyone can encrypt with public key, only private key can decrypt
```

### AES Encryption
```python
from cryptography.aes import aes_encrypt, aes_decrypt, generate_aes_key

# Generate secure key
key = generate_aes_key()  # 256-bit key
print(f"AES Key: {key.hex()}")

# Encrypt data
plaintext = b"This is a secret message that needs to be encrypted securely"
ciphertext, iv = aes_encrypt(plaintext, key)
print(f"Encrypted: {ciphertext.hex()}")
print(f"IV: {iv.hex()}")

# Decrypt data
decrypted = aes_decrypt(ciphertext, key, iv)
print(f"Decrypted: {decrypted.decode()}")
```

## Security Analysis

### Caesar Cipher
- **Vulnerability**: Frequency analysis, brute force (only 25 keys)
- **Attack Methods**: Letter frequency matching, exhaustive key search
- **Cryptanalysis Complexity**: O(1) - trivial to break
- **Historical Use**: Ancient Rome, simple military communications

### Vigenère Cipher
- **Vulnerability**: Kasiski examination, Index of Coincidence
- **Attack Methods**: Finding keyword length, then frequency analysis
- **Cryptanalysis Complexity**: O(k × 26^k) where k is keyword length
- **Historical Use**: 16th-19th century diplomatic communications

### RSA
- **Security Basis**: Difficulty of factoring large composite numbers
- **Key Size Recommendations**: 
  - 1024 bits: Deprecated
  - 2048 bits: Current minimum
  - 4096 bits: Future-proof
- **Attack Resistance**: No known efficient quantum attacks (except Shor's algorithm)
- **Implementation Considerations**: Padding schemes, side-channel attacks

### AES
- **Security Level**: No practical attacks known
- **Key Sizes**:
  - AES-128: ~128 bits of security
  - AES-192: ~192 bits of security
  - AES-256: ~256 bits of security
- **Quantum Resistance**: Grover's algorithm reduces effective key size by half

## Algorithm Complexity

### Time Complexity
| Algorithm | Encryption | Decryption | Key Generation |
|-----------|------------|------------|----------------|
| Caesar | O(n) | O(n) | O(1) |
| Vigenère | O(n) | O(n) | O(1) |
| RSA | O(k³) | O(k³) | O(k⁴) |
| AES | O(n) | O(n) | O(1) |

*Where n is message length, k is key size in bits*

### Space Complexity
- **Caesar/Vigenère**: O(1) auxiliary space
- **RSA**: O(k) for key storage
- **AES**: O(1) auxiliary space for encryption/decryption

## Educational Value

### Classical Ciphers (Caesar, Vigenère)
- **Learning Objectives**:
  - Understanding substitution ciphers
  - Frequency analysis techniques
  - Evolution of cryptographic methods
  - Weaknesses of simple encryption

### Modern Cryptography (RSA, AES)
- **Learning Objectives**:
  - Public-key vs symmetric cryptography
  - Mathematical foundations of security
  - Key management principles
  - Performance vs security tradeoffs

## Important Security Notes

1. **Educational Purpose**: These implementations are for learning and should not be used in production
2. **Side-Channel Attacks**: Real implementations must consider timing attacks, power analysis
3. **Random Number Generation**: Cryptographic security requires cryptographically secure random numbers
4. **Key Management**: Secure key generation, distribution, and storage are critical
5. **Protocol Security**: Encryption alone doesn't guarantee security; secure protocols are essential

## Real-World Applications

### Caesar/Vigenère Ciphers
- ROT13 text obfuscation
- Puzzle games and competitions
- Historical cryptanalysis education

### RSA
- HTTPS/TLS certificate signatures
- Email encryption (PGP/GPG)
- Code signing
- Cryptocurrency transactions

### AES
- File and disk encryption
- VPN tunnels
- Wireless security (WPA2/WPA3)
- Database encryption
- Secure messaging applications
