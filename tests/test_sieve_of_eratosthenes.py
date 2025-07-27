import unittest
from mathematical.sieve_of_eratosthenes import sieve_of_eratosthenes

class TestSieveOfEratosthenes(unittest.TestCase):
    def test_primes_up_to_10(self):
        self.assertEqual(sieve_of_eratosthenes(10), [2, 3, 5, 7])

    def test_primes_up_to_1(self):
        self.assertEqual(sieve_of_eratosthenes(1), [])

    def test_primes_up_to_20(self):
        self.assertEqual(sieve_of_eratosthenes(20), [2, 3, 5, 7, 11, 13, 17, 19])

if __name__ == "__main__":
    unittest.main()
