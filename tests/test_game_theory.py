"""
Test cases for Game Theory algorithms
"""

import pytest
from game_theory.minimax import minimax, alpha_beta_pruning, TicTacToe
from game_theory.nash_equilibrium import (
    find_nash_equilibrium, is_nash_equilibrium, prisoners_dilemma,
    battle_of_sexes, matching_pennies, find_mixed_strategy_equilibrium_2x2
)
from game_theory.auction import (
    first_price_auction, second_price_auction, english_auction,
    dutch_auction, vcg_auction, CombinatoralAuction
)

class TestMinimax:
    def test_tic_tac_toe_basic(self):
        """Test basic TicTacToe functionality."""
        game = TicTacToe()
        
        # Test initial state
        assert not game.is_terminal()
        assert game.current_player == 'X'
        
        # Test making moves
        assert game.make_move(0, 0)
        assert game.board[0][0] == 'X'
        assert game.current_player == 'O'
        
        # Test invalid move
        assert not game.make_move(0, 0)
        
        # Test winner detection
        game.board = [['X', 'X', 'X'], [' ', ' ', ' '], [' ', ' ', ' ']]
        assert game.is_winner('X')
        assert not game.is_winner('O')
    
    def test_minimax_simple(self):
        """Test minimax with simple evaluation function."""
        def get_children(x):
            return [x + 1, x + 2] if x < 5 else []
        
        def evaluate(x):
            return x
        
        def is_terminal(x):
            return x >= 5
        
        score, move = minimax(1, 3, True, get_children, evaluate, is_terminal)
        assert score == 6  # Should choose path that maximizes
        assert move == 2

class TestNashEquilibrium:
    def test_is_nash_equilibrium(self):
        """Test Nash equilibrium detection."""
        payoff_matrices, _, _ = prisoners_dilemma()
        
        # (Defect, Defect) should be Nash equilibrium
        assert is_nash_equilibrium(payoff_matrices, [1, 1])
        
        # (Cooperate, Cooperate) should not be Nash equilibrium
        assert not is_nash_equilibrium(payoff_matrices, [0, 0])
    
    def test_find_nash_equilibrium(self):
        """Test finding all Nash equilibria."""
        payoff_matrices, _, _ = prisoners_dilemma()
        equilibria = find_nash_equilibrium(payoff_matrices)
        
        # Should find (Defect, Defect) as the only pure strategy equilibrium
        assert len(equilibria) == 1
        assert equilibria[0][0] == [1, 1]
        assert equilibria[0][1] == [1, 1]  # Payoffs should be (1, 1)
    
    def test_battle_of_sexes(self):
        """Test Battle of the Sexes game."""
        payoff_matrices, _, _ = battle_of_sexes()
        equilibria = find_nash_equilibrium(payoff_matrices)
        
        # Should have two pure strategy equilibria
        assert len(equilibria) == 2
        strategies = [eq[0] for eq in equilibria]
        assert [0, 0] in strategies  # (Opera, Opera)
        assert [1, 1] in strategies  # (Football, Football)
    
    def test_mixed_strategy_equilibrium(self):
        """Test mixed strategy equilibrium for 2x2 games."""
        payoff_matrices, _, _ = matching_pennies()
        mixed_eq = find_mixed_strategy_equilibrium_2x2(payoff_matrices)
        
        # Should have mixed strategy equilibrium at (0.5, 0.5) for both players
        assert mixed_eq is not None
        p1_probs, p2_probs = mixed_eq
        assert abs(p1_probs[0] - 0.5) < 1e-6
        assert abs(p1_probs[1] - 0.5) < 1e-6
        assert abs(p2_probs[0] - 0.5) < 1e-6
        assert abs(p2_probs[1] - 0.5) < 1e-6

class TestAuctions:
    def test_first_price_auction(self):
        """Test first-price auction."""
        bids = [("Alice", 100), ("Bob", 150), ("Charlie", 120)]
        winner, winning_bid, payment = first_price_auction(bids)
        
        assert winner == "Bob"
        assert winning_bid == 150
        assert payment == 150
    
    def test_second_price_auction(self):
        """Test second-price (Vickrey) auction."""
        bids = [("Alice", 100), ("Bob", 150), ("Charlie", 120)]
        winner, winning_bid, payment = second_price_auction(bids)
        
        assert winner == "Bob"
        assert winning_bid == 150
        assert payment == 120  # Second highest bid
    
    def test_english_auction(self):
        """Test English auction."""
        bidders = ["Alice", "Bob", "Charlie"]
        valuations = [100, 150, 120]
        winner, final_price = english_auction(bidders, valuations, 10)
        
        assert winner == "Bob"
        assert final_price == 130  # Should stop when Charlie drops out
    
    def test_dutch_auction(self):
        """Test Dutch auction."""
        bidders = ["Alice", "Bob", "Charlie"]
        valuations = [100, 150, 120]
        winner, final_price = dutch_auction(bidders, valuations, 200, 10)
        
        assert winner == "Bob"
        assert final_price <= 150  # Bob should win at his valuation or below
    
    def test_combinatorial_auction(self):
        """Test combinatorial auction."""
        items = ["A", "B", "C"]
        auction = CombinatoralAuction(items)
        
        auction.add_bid("Alice", ["A", "B"], 100)
        auction.add_bid("Bob", ["B", "C"], 80)
        auction.add_bid("Charlie", ["A"], 60)
        
        winners = auction.solve_winner_determination()
        
        # Should allocate efficiently
        assert len(winners) > 0
        total_value = sum(bid for _, _, bid in winners)
        assert total_value > 0
