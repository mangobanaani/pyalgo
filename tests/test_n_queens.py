import unittest
from backtracking.n_queens import solve_n_queens

class TestNQueens(unittest.TestCase):
    def test_n_queens_4(self):
        """Test the N-Queens solver for n=4."""
        solutions = solve_n_queens(4)
        expected = [
            [".Q..", "...Q", "Q...", "..Q."],
            ["..Q.", "Q...", "...Q", ".Q.."]
        ]
        self.assertEqual(len(solutions), len(expected))
        self.assertTrue(all(sol in expected for sol in solutions))

    def test_n_queens_1(self):
        """Test the N-Queens solver for n=1."""
        solutions = solve_n_queens(1)
        expected = [["Q"]]
        self.assertEqual(solutions, expected)

    def test_n_queens_no_solution(self):
        """Test the N-Queens solver for n=2 and n=3 (no solutions)."""
        self.assertEqual(solve_n_queens(2), [])
        self.assertEqual(solve_n_queens(3), [])

if __name__ == "__main__":
    unittest.main()
