"""
Maze Solver

This module provides a solution to the Maze problem using backtracking.
"""

def solve_maze(maze):
    """
    Solves the Maze problem using backtracking.

    Args:
        maze (List[List[int]]): A 2D grid representing the maze.
        0 represents a wall, and 1 represents a path.

    Returns:
        List[List[int]]: A solution to the maze, where 1 represents the path taken.
        Returns an empty list if no solution exists.
    """
    def is_valid_move(x, y):
        return 0 <= x < n and 0 <= y < n and maze[x][y] == 1 and solution[x][y] == 0

    def backtrack(x, y):
        if x == n - 1 and y == n - 1:  # Reached the destination
            solution[x][y] = 1
            return True

        if is_valid_move(x, y):
            solution[x][y] = 1

            for dx, dy in moves:
                if backtrack(x + dx, y + dy):
                    return True

            solution[x][y] = 0

        return False

    n = len(maze)
    solution = [[0 for _ in range(n)] for _ in range(n)]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    if backtrack(0, 0):
        return solution
    return []
