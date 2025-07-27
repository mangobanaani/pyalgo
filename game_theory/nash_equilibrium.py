"""
Nash Equilibrium detection and computation.

This module implements algorithms for finding Nash equilibria in game theory,
particularly for 2-player games represented in normal form.
"""

from typing import List, Tuple, Optional, Dict
import itertools


def is_nash_equilibrium(payoff_matrices: List[List[List[int]]], strategies: List[int]) -> bool:
    """
    Check if given strategies form a Nash equilibrium.
    
    Args:
        payoff_matrices: List of payoff matrices for each player
        strategies: Strategy indices for each player
        
    Returns:
        True if strategies form Nash equilibrium
        
    Time Complexity: O(n * m) where n is number of players, m is max strategies per player
    """
    num_players = len(payoff_matrices)
    
    for player in range(num_players):
        current_payoff = _get_payoff(payoff_matrices[player], strategies)
        
        # Check if player can improve by changing strategy
        for alt_strategy in range(len(payoff_matrices[player])):
            if alt_strategy == strategies[player]:
                continue
                
            alt_strategies = strategies.copy()
            alt_strategies[player] = alt_strategy
            alt_payoff = _get_payoff(payoff_matrices[player], alt_strategies)
            
            if alt_payoff > current_payoff:
                return False  # Player can improve, not Nash equilibrium
    
    return True


def _get_payoff(payoff_matrix: List[List[int]], strategies: List[int]) -> int:
    """Get payoff from matrix given strategy profile."""
    result = payoff_matrix
    for strategy in strategies:
        result = result[strategy]
    return result


def find_nash_equilibrium(payoff_matrices: List[List[List[int]]]) -> List[Tuple[List[int], List[int]]]:
    """
    Find all pure strategy Nash equilibria in a game.
    
    Args:
        payoff_matrices: List of payoff matrices for each player
        
    Returns:
        List of (strategy_profile, payoffs) tuples representing equilibria
        
    Time Complexity: O(S^n * n * S) where S is max strategies, n is players
    """
    if not payoff_matrices:
        return []
    
    num_players = len(payoff_matrices)
    strategy_counts = [len(matrix) for matrix in payoff_matrices]
    
    equilibria = []
    
    # Generate all possible strategy profiles
    for strategies in itertools.product(*[range(count) for count in strategy_counts]):
        strategies = list(strategies)
        
        if is_nash_equilibrium(payoff_matrices, strategies):
            payoffs = [_get_payoff(payoff_matrices[i], strategies) for i in range(num_players)]
            equilibria.append((strategies, payoffs))
    
    return equilibria


def prisoners_dilemma() -> Tuple[List[List[List[int]]], List[str], List[str]]:
    """
    Create the classic Prisoner's Dilemma game.
    
    Returns:
        Tuple of (payoff_matrices, player1_actions, player2_actions)
    """
    # Actions: 0 = Cooperate, 1 = Defect
    actions = ["Cooperate", "Defect"]
    
    # Player 1's payoff matrix
    payoff_p1 = [
        [3, 0],  # P1 cooperates: (C,C)=3, (C,D)=0
        [5, 1]   # P1 defects: (D,C)=5, (D,D)=1
    ]
    
    # Player 2's payoff matrix  
    payoff_p2 = [
        [3, 5],  # P2 cooperates: (C,C)=3, (D,C)=5
        [0, 1]   # P2 defects: (C,D)=0, (D,D)=1
    ]
    
    return [payoff_p1, payoff_p2], actions, actions


def battle_of_sexes() -> Tuple[List[List[List[int]]], List[str], List[str]]:
    """
    Create the Battle of the Sexes game.
    
    Returns:
        Tuple of (payoff_matrices, player1_actions, player2_actions)
    """
    # Actions: 0 = Opera, 1 = Football
    actions = ["Opera", "Football"]
    
    payoff_p1 = [
        [2, 0],  # P1 chooses Opera: (O,O)=2, (O,F)=0
        [0, 1]   # P1 chooses Football: (F,O)=0, (F,F)=1
    ]
    
    payoff_p2 = [
        [1, 0],  # P2 chooses Opera: (O,O)=1, (F,O)=0
        [0, 2]   # P2 chooses Football: (O,F)=0, (F,F)=2
    ]
    
    return [payoff_p1, payoff_p2], actions, actions


def matching_pennies() -> Tuple[List[List[List[int]]], List[str], List[str]]:
    """
    Create the Matching Pennies game.
    
    Returns:
        Tuple of (payoff_matrices, player1_actions, player2_actions)
    """
    # Actions: 0 = Heads, 1 = Tails
    actions = ["Heads", "Tails"]
    
    payoff_p1 = [
        [1, -1],  # P1 Heads: (H,H)=1, (H,T)=-1
        [-1, 1]   # P1 Tails: (T,H)=-1, (T,T)=1
    ]
    
    payoff_p2 = [
        [-1, 1],  # P2 Heads: (H,H)=-1, (T,H)=1
        [1, -1]   # P2 Tails: (H,T)=1, (T,T)=-1
    ]
    
    return [payoff_p1, payoff_p2], actions, actions


def find_mixed_strategy_equilibrium_2x2(payoff_matrices: List[List[List[int]]]) -> Optional[Tuple[List[float], List[float]]]:
    """
    Find mixed strategy Nash equilibrium for 2x2 games.
    
    Uses the indifference principle to compute mixed strategies.
    
    Args:
        payoff_matrices: 2x2 payoff matrices for both players
        
    Returns:
        Tuple of mixed strategies (probabilities) or None if no mixed equilibrium
        
    Time Complexity: O(1)
    """
    if len(payoff_matrices) != 2 or any(len(matrix) != 2 or len(matrix[0]) != 2 for matrix in payoff_matrices):
        return None
    
    A, B = payoff_matrices
    
    # For player 1: find probability p that makes player 2 indifferent
    # Player 2's expected payoffs for each action:
    # Action 0: p * B[0][0] + (1-p) * B[1][0]
    # Action 1: p * B[0][1] + (1-p) * B[1][1]
    # Set equal: p * B[0][0] + (1-p) * B[1][0] = p * B[0][1] + (1-p) * B[1][1]
    
    denom_p1 = (B[0][0] - B[1][0]) - (B[0][1] - B[1][1])
    if denom_p1 == 0:
        return None  # No mixed strategy equilibrium
    
    p = (B[1][1] - B[1][0]) / denom_p1
    
    # For player 2: find probability q that makes player 1 indifferent
    denom_p2 = (A[0][0] - A[0][1]) - (A[1][0] - A[1][1])
    if denom_p2 == 0:
        return None
    
    q = (A[1][1] - A[0][1]) / denom_p2
    
    # Check if probabilities are valid (between 0 and 1)
    if 0 <= p <= 1 and 0 <= q <= 1:
        return ([p, 1-p], [q, 1-q])
    
    return None


def analyze_game(payoff_matrices: List[List[List[int]]], 
                action_names: List[List[str]] = None) -> Dict:
    """
    Comprehensive analysis of a game.
    
    Args:
        payoff_matrices: Payoff matrices for all players
        action_names: Optional names for actions
        
    Returns:
        Dictionary with game analysis results
    """
    analysis = {}
    
    # Find pure strategy Nash equilibria
    pure_equilibria = find_nash_equilibrium(payoff_matrices)
    analysis['pure_nash_equilibria'] = pure_equilibria
    
    # For 2x2 games, also find mixed strategy equilibrium
    if (len(payoff_matrices) == 2 and 
        all(len(matrix) == 2 and len(matrix[0]) == 2 for matrix in payoff_matrices)):
        
        mixed_eq = find_mixed_strategy_equilibrium_2x2(payoff_matrices)
        analysis['mixed_nash_equilibrium'] = mixed_eq
    
    # Game properties
    analysis['num_players'] = len(payoff_matrices)
    analysis['strategy_counts'] = [len(matrix) for matrix in payoff_matrices]
    
    # Action names
    if action_names:
        analysis['action_names'] = action_names
    
    return analysis


def dominant_strategies(payoff_matrix: List[List[int]]) -> Dict[str, List[int]]:
    """
    Find dominant strategies for a player.
    
    Args:
        payoff_matrix: Player's payoff matrix
        
    Returns:
        Dictionary with 'strictly_dominant' and 'weakly_dominant' strategies
    """
    num_strategies = len(payoff_matrix)
    strictly_dominant = []
    weakly_dominant = []
    
    for i in range(num_strategies):
        is_strictly_dominant = True
        is_weakly_dominant = True
        
        for j in range(num_strategies):
            if i == j:
                continue
                
            # Compare strategy i with strategy j across all opponent strategies
            strict_better = True
            weak_better = True
            
            for k in range(len(payoff_matrix[0])):
                if payoff_matrix[i][k] <= payoff_matrix[j][k]:
                    strict_better = False
                if payoff_matrix[i][k] < payoff_matrix[j][k]:
                    weak_better = False
            
            if not strict_better:
                is_strictly_dominant = False
            if not weak_better:
                is_weakly_dominant = False
        
        if is_strictly_dominant:
            strictly_dominant.append(i)
        elif is_weakly_dominant:
            weakly_dominant.append(i)
    
    return {
        'strictly_dominant': strictly_dominant,
        'weakly_dominant': weakly_dominant
    }
