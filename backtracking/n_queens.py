"""
N-Queens Problem Solver

This module provides a solution to the N-Queens problem using backtracking.
"""

def solve_n_queens(n):
    """
    Solves the N-Queens problem for a given board size n.

    Args:
        n (int): The size of the chessboard (n x n).

    Returns:
        List[List[str]]: A list of solutions, where each solution is represented
        as a list of strings. Each string represents a row of the chessboard,
        with 'Q' for a queen and '.' for an empty space.
    """
    def is_safe(row, col):
        return col not in cols and (row - col) not in diag1 and (row + col) not in diag2

    def place_queen(row):
        if row == n:
            solutions.append(["".join(board[r]) for r in range(n)])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)

                place_queen(row + 1)

                board[row][col] = '.'
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)

    solutions = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    cols, diag1, diag2 = set(), set(), set()
    place_queen(0)
    return solutions
