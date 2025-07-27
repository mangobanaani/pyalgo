import unittest
from backtracking.crossword_solver import solve_crossword

class TestCrosswordSolver(unittest.TestCase):
    def test_crossword_solver(self):
        """Test the Crossword solver with a valid puzzle."""
        grid = [
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-']
        ]

        words = ["hello", "world"]

        self.assertTrue(solve_crossword(grid, words))

    def test_crossword_no_solution(self):
        """Test the Crossword solver with an unsolvable puzzle."""
        grid = [
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-'],
            ['-', '-', '-', '-', '-']
        ]

        words = ["hello", "world", "python"]

        self.assertFalse(solve_crossword(grid, words))

if __name__ == "__main__":
    unittest.main()
