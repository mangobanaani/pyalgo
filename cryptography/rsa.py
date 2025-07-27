"""
RSA (Rivest–Shamir–Adleman) Algorithm

A public-key cryptosystem that is widely used for secure data transmission.
It relies on the practical difficulty of factoring the product of two large prime numbers.
"""
import random
from math import gcd

def is_prime(n, k=5):
    """
    Miller-Rabin primality test.
    
    :param n: Number to test for primality
    :param k: Number of rounds of testing
    :return: Boolean indicating whether n is probably prime
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write n as 2^r * d + 1
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    # Witness loop
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    """
    Generate a random prime number with the specified number of bits.
    
    :param bits: Number of bits for the prime
    :return: A prime number
    """
    while True:
        # Generate a random odd number with the specified number of bits
        p = random.getrandbits(bits) | (1 << bits - 1) | 1
        if is_prime(p):
            return p

def extended_gcd(a, b):
    """
    Extended Euclidean algorithm to find the greatest common divisor and coefficients.
    
    :param a: First number
    :param b: Second number
    :return: Tuple of (gcd, x, y) such that ax + by = gcd
    """
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x

def mod_inverse(a, m):
    """
    Calculate the modular multiplicative inverse of a under modulo m.
    
    :param a: Number to find inverse for
    :param m: Modulus
    :return: Modular multiplicative inverse
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

def generate_key_pair(bits=1024):
    """
    Generate an RSA key pair (public and private keys).
    
    :param bits: Number of bits for each prime number
    :return: Tuple of ((e, n), (d, n)) representing public and private keys
    """
    # Generate two random prime numbers
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)
    
    # Calculate n = p * q
    n = p * q
    
    # Calculate Euler's totient function: φ(n) = (p-1)(q-1)
    phi = (p - 1) * (q - 1)
    
    # Choose e such that 1 < e < φ(n) and gcd(e, φ(n)) = 1
    e = 65537  # Common choice for e
    
    # Ensure e and φ(n) are coprime
    while gcd(e, phi) != 1:
        e = random.randrange(2, phi)
    
    # Calculate d, the modular multiplicative inverse of e (mod φ(n))
    d = mod_inverse(e, phi)
    
    return ((e, n), (d, n))

def rsa_encrypt(plaintext, public_key):
    """
    Encrypt a message using RSA algorithm.
    
    :param plaintext: Message as an integer
    :param public_key: Public key as tuple (e, n)
    :return: Encrypted message
    """
    e, n = public_key
    return pow(plaintext, e, n)

def rsa_decrypt(ciphertext, private_key):
    """
    Decrypt a message using RSA algorithm.
    
    :param ciphertext: Encrypted message
    :param private_key: Private key as tuple (d, n)
    :return: Decrypted message as an integer
    """
    d, n = private_key
    return pow(ciphertext, d, n)

def string_to_int(message):
    """
    Convert a string to an integer.
    
    :param message: String message
    :return: Integer representation
    """
    return int.from_bytes(message.encode(), 'big')

def int_to_string(message_int):
    """
    Convert an integer back to a string.
    
    :param message_int: Integer representation
    :return: Original string
    """
    return message_int.to_bytes((message_int.bit_length() + 7) // 8, 'big').decode()

def encrypt_string(message, public_key):
    """
    Encrypt a string message using RSA.
    
    :param message: String message to encrypt
    :param public_key: Public key as tuple (e, n)
    :return: Encrypted message as an integer
    """
    # Convert the string to an integer
    m = string_to_int(message)
    
    # Check if the message is too long for the key
    _, n = public_key
    if m >= n:
        raise ValueError("Message is too large for the key")
    
    # Encrypt the integer
    return rsa_encrypt(m, public_key)

def decrypt_string(ciphertext, private_key):
    """
    Decrypt an encrypted message to recover the original string.
    
    :param ciphertext: Encrypted message
    :param private_key: Private key as tuple (d, n)
    :return: Decrypted string message
    """
    # Decrypt to get the integer
    m = rsa_decrypt(ciphertext, private_key)
    
    # Convert the integer back to a string
    return int_to_string(m)
