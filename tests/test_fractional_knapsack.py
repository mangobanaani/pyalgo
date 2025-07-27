import unittest
from greedy.fractional_knapsack import Item, fractional_knapsack

class TestFractionalKnapsack(unittest.TestCase):
    def test_fractional_knapsack(self):
        items = [
            Item(60, 10),  # value: 60, weight: 10
            Item(100, 20), # value: 100, weight: 20
            Item(120, 30)  # value: 120, weight: 30
        ]
        capacity = 50
        result = fractional_knapsack(capacity, items)
        self.assertAlmostEqual(result, 240.0, places=2)

    def test_empty_knapsack(self):
        items = []
        capacity = 50
        result = fractional_knapsack(capacity, items)
        self.assertEqual(result, 0.0)

    def test_zero_capacity(self):
        items = [
            Item(60, 10),
            Item(100, 20),
            Item(120, 30)
        ]
        capacity = 0
        result = fractional_knapsack(capacity, items)
        self.assertEqual(result, 0.0)

if __name__ == "__main__":
    unittest.main()
