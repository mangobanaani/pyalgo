import unittest
from mathematical.gcd import gcd

class TestGCD(unittest.TestCase):
    def test_gcd_of_48_and_18(self):
        self.assertEqual(gcd(48, 18), 6)

    def test_gcd_of_101_and_103(self):
        self.assertEqual(gcd(101, 103), 1)

    def test_gcd_of_0_and_5(self):
        self.assertEqual(gcd(0, 5), 5)

    def test_gcd_of_0_and_0(self):
        self.assertEqual(gcd(0, 0), 0)

if __name__ == "__main__":
    unittest.main()
