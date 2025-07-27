"""
Sudoku Solver

This module provides a solution to the Sudoku puzzle using backtracking.
"""

def solve_sudoku(board):
    """
    Solves the Sudoku puzzle using backtracking.

    Args:
        board (List[List[int]]): A 9x9 grid representing the Sudoku puzzle.
        Empty cells are represented by 0.

    Returns:
        bool: True if the puzzle is solved, False otherwise.
    """
    def is_valid(num, row, col):
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[r][col] for r in range(9)]:
            return False

        # Check 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if is_valid(num, row, col):
                            board[row][col] = num

                            if backtrack():
                                return True

                            board[row][col] = 0

                    return False

        return True

    return backtrack()
