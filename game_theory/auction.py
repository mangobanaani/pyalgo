"""
Auction Algorithms

Implementations of various auction mechanisms and their analysis.
"""

from typing import List, Tuple, Dict, Any
import heapq

def first_price_auction(bids: List[Tuple[str, float]]) -> Tuple[str, float, float]:
    """
    First-price sealed-bid auction.
    
    Args:
        bids: List of (bidder_id, bid_amount) tuples
        
    Returns:
        Tuple of (winner_id, winning_bid, payment)
    """
    if not bids:
        raise ValueError("No bids provided")
    
    # Find the highest bid
    winner_id, winning_bid = max(bids, key=lambda x: x[1])
    payment = winning_bid
    
    return winner_id, winning_bid, payment

def second_price_auction(bids: List[Tuple[str, float]]) -> Tuple[str, float, float]:
    """
    Second-price sealed-bid auction (Vickrey auction).
    
    Args:
        bids: List of (bidder_id, bid_amount) tuples
        
    Returns:
        Tuple of (winner_id, winning_bid, payment)
    """
    if not bids:
        raise ValueError("No bids provided")
    
    if len(bids) == 1:
        winner_id, winning_bid = bids[0]
        return winner_id, winning_bid, 0.0
    
    # Sort bids in descending order
    sorted_bids = sorted(bids, key=lambda x: x[1], reverse=True)
    
    winner_id, winning_bid = sorted_bids[0]
    second_highest_bid = sorted_bids[1][1]
    
    return winner_id, winning_bid, second_highest_bid

def english_auction(bidders: List[str], valuations: List[float], increment: float = 1.0) -> Tuple[str, float]:
    """
    Simulate an English auction (ascending bid auction).
    
    Args:
        bidders: List of bidder IDs
        valuations: List of private valuations for each bidder
        increment: Minimum bid increment
        
    Returns:
        Tuple of (winner_id, final_price)
    """
    if len(bidders) != len(valuations):
        raise ValueError("Number of bidders must match number of valuations")
    
    if not bidders:
        raise ValueError("No bidders provided")
    
    # Create list of active bidders with their valuations
    active_bidders = list(zip(bidders, valuations))
    current_price = 0.0
    
    while len(active_bidders) > 1:
        current_price += increment
        
        # Remove bidders whose valuation is below current price
        active_bidders = [(bidder, val) for bidder, val in active_bidders if val >= current_price]
    
    if active_bidders:
        winner_id = active_bidders[0][0]
        return winner_id, current_price
    else:
        # No one can afford the starting price
        return "", 0.0

def vcg_auction(bidders: List[str], valuations: Dict[str, List[float]], k: int = 1) -> List[Tuple[str, float]]:
    """
    Vickrey-Clarke-Groves (VCG) auction for multiple items.
    
    Args:
        bidders: List of bidder IDs
        valuations: Dictionary mapping bidder_id to list of valuations for each item
        k: Number of items to allocate
        
    Returns:
        List of (winner_id, payment) tuples
    """
    if not bidders or not valuations:
        return []
    
    # Calculate total value for each bidder across all items
    total_valuations = []
    for bidder in bidders:
        if bidder in valuations:
            total_val = sum(valuations[bidder])
            total_valuations.append((bidder, total_val))
    
    # Sort by total valuation (descending)
    total_valuations.sort(key=lambda x: x[1], reverse=True)
    
    # Select top k bidders as winners
    winners = total_valuations[:k]
    
    # Calculate VCG payments
    payments = []
    for winner_id, winner_val in winners:
        # Calculate what others would have received without this winner
        others_without_winner = [val for bidder, val in total_valuations 
                               if bidder != winner_id][:k-1]
        others_value_without = sum(others_without_winner)
        
        # Calculate what others actually receive with this winner
        others_with_winner = [val for bidder, val in total_valuations 
                            if bidder != winner_id][:k]
        others_value_with = sum(others_with_winner)
        
        # VCG payment is the externality imposed on others
        payment = others_value_without - others_value_with
        payments.append((winner_id, max(0, payment)))
    
    return payments

def dutch_auction(bidders: List[str], valuations: List[float], starting_price: float, 
                 decrement: float = 1.0) -> Tuple[str, float]:
    """
    Simulate a Dutch auction (descending price auction).
    
    Args:
        bidders: List of bidder IDs
        valuations: List of private valuations for each bidder
        starting_price: Starting price for the auction
        decrement: Price decrement per round
        
    Returns:
        Tuple of (winner_id, final_price)
    """
    if len(bidders) != len(valuations):
        raise ValueError("Number of bidders must match number of valuations")
    
    if not bidders:
        raise ValueError("No bidders provided")
    
    current_price = starting_price
    bidder_valuations = list(zip(bidders, valuations))
    
    while current_price > 0:
        # Check if any bidder is willing to buy at current price
        willing_bidders = [(bidder, val) for bidder, val in bidder_valuations 
                          if val >= current_price]
        
        if willing_bidders:
            # First bidder to accept wins (in practice, this would be random or first to respond)
            winner_id = max(willing_bidders, key=lambda x: x[1])[0]  # Highest valuation wins ties
            return winner_id, current_price
        
        current_price -= decrement
    
    return "", 0.0

class CombinatoralAuction:
    """
    Combinatorial auction where bidders can bid on packages of items.
    """
    
    def __init__(self, items: List[str]):
        self.items = items
        self.bids = []  # List of (bidder_id, item_set, bid_amount)
    
    def add_bid(self, bidder_id: str, item_set: List[str], bid_amount: float):
        """Add a bid for a set of items."""
        # Validate that all items in the set exist
        for item in item_set:
            if item not in self.items:
                raise ValueError(f"Item {item} not available")
        
        self.bids.append((bidder_id, set(item_set), bid_amount))
    
    def solve_winner_determination(self) -> List[Tuple[str, List[str], float]]:
        """
        Solve the winner determination problem (simplified greedy approach).
        
        Returns:
            List of (winner_id, items_won, bid_amount) tuples
        """
        if not self.bids:
            return []
        
        # Sort bids by value per item (greedy heuristic)
        sorted_bids = sorted(self.bids, key=lambda x: x[2] / len(x[1]), reverse=True)
        
        allocated_items = set()
        winners = []
        
        for bidder_id, item_set, bid_amount in sorted_bids:
            # Check if any items in this bid are already allocated
            if not item_set.intersection(allocated_items):
                # All items are available, allocate to this bidder
                allocated_items.update(item_set)
                winners.append((bidder_id, list(item_set), bid_amount))
        
        return winners
