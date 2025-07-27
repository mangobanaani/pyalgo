import unittest
from backtracking.knights_tour import solve_knights_tour

class TestKnightsTour(unittest.TestCase):
    def test_knights_tour_5x5(self):
        """Test the Knight's Tour solver for a 5x5 board."""
        solution = solve_knights_tour(5)
        self.assertTrue(len(solution) > 0)  # Ensure a solution exists
        self.assertEqual(len(solution), 5)
        self.assertEqual(len(solution[0]), 5)

    def test_knights_tour_no_solution(self):
        """Test the Knight's Tour solver for a 2x2 board (no solution)."""
        self.assertEqual(solve_knights_tour(2), [])

if __name__ == "__main__":
    unittest.main()
