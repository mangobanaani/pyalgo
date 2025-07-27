import unittest
from backtracking.maze_solver import solve_maze

class TestMazeSolver(unittest.TestCase):
    def test_maze_solver(self):
        """Test the Maze solver with a valid maze."""
        maze = [
            [1, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 1, 0, 0],
            [1, 1, 1, 1]
        ]

        expected_solution = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1]
        ]

        solution = solve_maze(maze)
        self.assertEqual(solution, expected_solution)

    def test_maze_no_solution(self):
        """Test the Maze solver with an unsolvable maze."""
        maze = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]

        self.assertEqual(solve_maze(maze), [])

if __name__ == "__main__":
    unittest.main()
