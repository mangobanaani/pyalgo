"""
Crossword Puzzle Solver

This module provides a solution to the Crossword Puzzle problem using backtracking.
"""

def solve_crossword(grid, words):
    """
    Solves the crossword puzzle by placing words into the grid.

    Args:
        grid (List[List[str]]): A 2D grid representing the crossword puzzle.
        Empty cells are represented by '-'.
        words (List[str]): A list of words to place in the grid.

    Returns:
        bool: True if the puzzle is solved, False otherwise.
    """
    def can_place_word_horizontally(row, col, word):
        if col + len(word) > len(grid[0]):
            return False

        for i in range(len(word)):
            if grid[row][col + i] not in ('-', word[i]):
                return False

        return True

    def can_place_word_vertically(row, col, word):
        if row + len(word) > len(grid):
            return False

        for i in range(len(word)):
            if grid[row + i][col] not in ('-', word[i]):
                return False

        return True

    def place_word_horizontally(row, col, word):
        original = []
        for i in range(len(word)):
            original.append(grid[row][col + i])
            grid[row][col + i] = word[i]
        return original

    def place_word_vertically(row, col, word):
        original = []
        for i in range(len(word)):
            original.append(grid[row + i][col])
            grid[row + i][col] = word[i]
        return original

    def remove_word_horizontally(row, col, original):
        for i in range(len(original)):
            grid[row][col + i] = original[i]

    def remove_word_vertically(row, col, original):
        for i in range(len(original)):
            grid[row + i][col] = original[i]

    def backtrack(index):
        if index == len(words):
            return True

        word = words[index]

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if can_place_word_horizontally(row, col, word):
                    original = place_word_horizontally(row, col, word)
                    if backtrack(index + 1):
                        return True
                    remove_word_horizontally(row, col, original)

                if can_place_word_vertically(row, col, word):
                    original = place_word_vertically(row, col, word)
                    if backtrack(index + 1):
                        return True
                    remove_word_vertically(row, col, original)

        return False

    return backtrack(0)
