"""
Minimax Algorithm with Alpha-Beta Pruning

Used in two-player zero-sum games to find the optimal move.
Common applications: Tic-tac-toe, Chess, Checkers, etc.
"""

import math
from typing import List, Tuple, Any, Optional, Callable

def minimax(node: Any, depth: int, maximizing_player: bool, 
           get_children: Callable, evaluate: Callable, is_terminal: Callable) -> Tuple[float, Any]:
    """
    Minimax algorithm to find the best move in a two-player zero-sum game.
    
    Args:
        node: Current game state
        depth: Maximum search depth
        maximizing_player: True if current player is maximizing, False if minimizing
        get_children: Function that returns list of possible moves/states from current node
        evaluate: Function that evaluates the score of a terminal node
        is_terminal: Function that checks if a node is terminal (game over)
        
    Returns:
        Tuple of (best_score, best_move)
    """
    if depth == 0 or is_terminal(node):
        return evaluate(node), None
    
    children = get_children(node)
    
    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        
        for child in children:
            eval_score, _ = minimax(child, depth - 1, False, get_children, evaluate, is_terminal)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = child
                
        return max_eval, best_move
    else:
        min_eval = math.inf
        best_move = None
        
        for child in children:
            eval_score, _ = minimax(child, depth - 1, True, get_children, evaluate, is_terminal)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = child
                
        return min_eval, best_move

def alpha_beta_pruning(node: Any, depth: int, alpha: float, beta: float, 
                      maximizing_player: bool, get_children: Callable, 
                      evaluate: Callable, is_terminal: Callable) -> Tuple[float, Any]:
    """
    Minimax with Alpha-Beta pruning for improved efficiency.
    
    Args:
        node: Current game state
        depth: Maximum search depth
        alpha: Best already explored option along path to root for maximizer
        beta: Best already explored option along path to root for minimizer
        maximizing_player: True if current player is maximizing
        get_children: Function that returns list of possible moves/states
        evaluate: Function that evaluates terminal nodes
        is_terminal: Function that checks if node is terminal
        
    Returns:
        Tuple of (best_score, best_move)
    """
    if depth == 0 or is_terminal(node):
        return evaluate(node), None
    
    children = get_children(node)
    
    if maximizing_player:
        max_eval = -math.inf
        best_move = None
        
        for child in children:
            eval_score, _ = alpha_beta_pruning(child, depth - 1, alpha, beta, False, 
                                             get_children, evaluate, is_terminal)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = child
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
                
        return max_eval, best_move
    else:
        min_eval = math.inf
        best_move = None
        
        for child in children:
            eval_score, _ = alpha_beta_pruning(child, depth - 1, alpha, beta, True,
                                             get_children, evaluate, is_terminal)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = child
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
                
        return min_eval, best_move


class TicTacToe:
    """
    Simple Tic-Tac-Toe implementation for demonstrating minimax.
    """
    
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
    
    def make_move(self, row: int, col: int) -> bool:
        """Make a move if valid."""
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves."""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves
    
    def is_winner(self, player: str) -> bool:
        """Check if player has won."""
        # Check rows
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        
        # Check columns
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        
        # Check diagonals
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def is_full(self) -> bool:
        """Check if board is full."""
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.is_winner('X') or self.is_winner('O') or self.is_full()
    
    def evaluate(self) -> float:
        """Evaluate the current board state."""
        if self.is_winner('X'):
            return 1.0
        elif self.is_winner('O'):
            return -1.0
        else:
            return 0.0
    
    def copy(self):
        """Create a copy of the game state."""
        new_game = TicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game
    
    def get_children_states(self):
        """Get all possible next states."""
        children = []
        for row, col in self.get_valid_moves():
            child = self.copy()
            child.make_move(row, col)
            children.append(child)
        return children
    
    def get_best_move(self, depth: int = 9) -> Optional[Tuple[int, int]]:
        """Get the best move using minimax."""
        is_maximizing = self.current_player == 'X'
        
        def get_children(state):
            return state.get_children_states()
        
        def evaluate(state):
            return state.evaluate()
        
        def is_terminal(state):
            return state.is_terminal()
        
        _, best_state = minimax(self, depth, is_maximizing, get_children, evaluate, is_terminal)
        
        if best_state is None:
            return None
        
        # Find the move that leads to best_state
        for row, col in self.get_valid_moves():
            child = self.copy()
            child.make_move(row, col)
            if child.board == best_state.board:
                return (row, col)
        
        return None
