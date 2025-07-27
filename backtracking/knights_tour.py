"""
Knight's Tour Solver

This module provides a solution to the Knight's Tour problem using backtracking.
"""

def solve_knights_tour(n):
    """
    Solves the Knight's Tour problem for an n x n chessboard.

    Args:
        n (int): The size of the chessboard.

    Returns:
        List[List[int]]: A solution to the Knight's Tour problem, where each cell
        contains the step number of the knight's move. Returns an empty list if
        no solution exists.
    """
    def is_valid_move(x, y):
        return 0 <= x < n and 0 <= y < n and board[x][y] == -1

    def backtrack(x, y, move_count):
        if move_count == n * n:
            return True

        for dx, dy in moves:
            next_x, next_y = x + dx, y + dy
            if is_valid_move(next_x, next_y):
                board[next_x][next_y] = move_count
                if backtrack(next_x, next_y, move_count + 1):
                    return True
                board[next_x][next_y] = -1

        return False

    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    board = [[-1 for _ in range(n)] for _ in range(n)]
    board[0][0] = 0  # Start from the top-left corner

    if backtrack(0, 0, 1):
        return board
    return []
