"""
Simulated Annealing Implementation

Metaheuristic optimization algorithm inspired by the annealing process
in metallurgy. Uses probabilistic acceptance of worse solutions to escape
local optima.
"""

import random
import math
from typing import Callable, List, Tuple, Optional, Any, Union


class SimulatedAnnealing:
    """
    Simulated Annealing optimization algorithm.
    
    Probabilistically accepts worse solutions based on temperature schedule
    to escape local optima and find global optimum.
    """
    
    def __init__(self, initial_temp: float = 1000.0, final_temp: float = 1e-8,
                 cooling_rate: float = 0.95, max_iterations: int = 10000,
                 max_no_improvement: int = 1000, random_state: Optional[int] = None):
        """
        Initialize Simulated Annealing.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Ending temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            max_iterations: Maximum number of iterations
            max_no_improvement: Max iterations without improvement before stopping
            random_state: Random seed for reproducibility
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.max_no_improvement = max_no_improvement
        self.random_state = random_state
        
        # Optimization tracking
        self.best_solution_: Optional[Any] = None
        self.best_cost_: float = float('inf')
        self.cost_history_: List[float] = []
        self.temperature_history_: List[float] = []
        self.current_iteration_: int = 0
        
        if random_state is not None:
            random.seed(random_state)
    
    def cooling_schedule(self, iteration: int, temperature: float) -> float:
        """
        Update temperature according to cooling schedule.
        
        Args:
            iteration: Current iteration
            temperature: Current temperature
            
        Returns:
            New temperature
        """
        return temperature * self.cooling_rate
    
    def exponential_cooling(self, iteration: int) -> float:
        """Exponential cooling schedule."""
        return self.initial_temp * (self.cooling_rate ** iteration)
    
    def linear_cooling(self, iteration: int) -> float:
        """Linear cooling schedule."""
        if iteration >= self.max_iterations:
            return self.final_temp
        
        slope = (self.final_temp - self.initial_temp) / self.max_iterations
        return self.initial_temp + slope * iteration
    
    def logarithmic_cooling(self, iteration: int) -> float:
        """Logarithmic cooling schedule."""
        if iteration == 0:
            return self.initial_temp
        return self.initial_temp / math.log(1 + iteration)
    
    def acceptance_probability(self, current_cost: float, new_cost: float, 
                             temperature: float) -> float:
        """
        Calculate probability of accepting new solution.
        
        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            temperature: Current temperature
            
        Returns:
            Acceptance probability
        """
        if new_cost < current_cost:
            return 1.0  # Always accept better solutions
        
        if temperature <= 0:
            return 0.0
        
        return math.exp(-(new_cost - current_cost) / temperature)
    
    def should_accept(self, current_cost: float, new_cost: float, 
                     temperature: float) -> bool:
        """
        Determine whether to accept new solution.
        
        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            temperature: Current temperature
            
        Returns:
            True if solution should be accepted
        """
        probability = self.acceptance_probability(current_cost, new_cost, temperature)
        return random.random() < probability
    
    def optimize(self, cost_function: Callable[[Any], float],
                initial_solution: Any,
                neighbor_function: Callable[[Any], Any]) -> Tuple[Any, float]:
        """
        Run simulated annealing optimization.
        
        Args:
            cost_function: Function to minimize
            initial_solution: Starting solution
            neighbor_function: Function to generate neighbor solutions
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        # Initialize
        current_solution = initial_solution
        current_cost = cost_function(current_solution)
        
        self.best_solution_ = current_solution
        self.best_cost_ = current_cost
        self.cost_history_ = [current_cost]
        self.temperature_history_ = [self.initial_temp]
        
        temperature = self.initial_temp
        no_improvement_count = 0
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            self.current_iteration_ = iteration
            
            # Generate neighbor solution
            new_solution = neighbor_function(current_solution)
            new_cost = cost_function(new_solution)
            
            # Accept or reject new solution
            if self.should_accept(current_cost, new_cost, temperature):
                current_solution = new_solution
                current_cost = new_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Update best solution
            if new_cost < self.best_cost_:
                self.best_solution_ = new_solution
                self.best_cost_ = new_cost
                no_improvement_count = 0
            
            # Track history
            self.cost_history_.append(current_cost)
            self.temperature_history_.append(temperature)
            
            # Update temperature
            temperature = self.cooling_schedule(iteration, temperature)
            
            # Check stopping criteria
            if temperature < self.final_temp:
                break
            
            if no_improvement_count >= self.max_no_improvement:
                break
        
        return self.best_solution_, self.best_cost_
    
    def get_statistics(self) -> dict:
        """Get optimization statistics."""
        return {
            'best_cost': self.best_cost_,
            'final_temperature': self.temperature_history_[-1] if self.temperature_history_ else None,
            'iterations': len(self.cost_history_),
            'improvement_rate': self._calculate_improvement_rate()
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement over optimization."""
        if len(self.cost_history_) < 2:
            return 0.0
        
        initial_cost = self.cost_history_[0]
        final_cost = self.cost_history_[-1]
        
        if initial_cost == 0:
            return 0.0
        
        return (initial_cost - final_cost) / initial_cost


class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    """
    Adaptive Simulated Annealing with dynamic parameter adjustment.
    
    Adjusts temperature and acceptance criteria based on search progress.
    """
    
    def __init__(self, adaptation_interval: int = 100, **kwargs):
        """
        Initialize Adaptive SA.
        
        Args:
            adaptation_interval: Iterations between parameter adaptations
            **kwargs: Other SA parameters
        """
        super().__init__(**kwargs)
        self.adaptation_interval = adaptation_interval
        self.acceptance_rates_: List[float] = []
    
    def adapt_parameters(self, iteration: int, acceptance_rate: float) -> None:
        """
        Adapt parameters based on acceptance rate.
        
        Args:
            iteration: Current iteration
            acceptance_rate: Recent acceptance rate
        """
        # Adjust cooling rate based on acceptance rate
        if acceptance_rate < 0.1:
            # Too few acceptances, slow down cooling
            self.cooling_rate = min(0.99, self.cooling_rate * 1.05)
        elif acceptance_rate > 0.9:
            # Too many acceptances, speed up cooling
            self.cooling_rate = max(0.8, self.cooling_rate * 0.95)
    
    def optimize(self, cost_function: Callable[[Any], float],
                initial_solution: Any,
                neighbor_function: Callable[[Any], Any]) -> Tuple[Any, float]:
        """Run adaptive simulated annealing optimization."""
        # Initialize
        current_solution = initial_solution
        current_cost = cost_function(current_solution)
        
        self.best_solution_ = current_solution
        self.best_cost_ = current_cost
        self.cost_history_ = [current_cost]
        self.temperature_history_ = [self.initial_temp]
        
        temperature = self.initial_temp
        acceptance_count = 0
        total_attempts = 0
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            self.current_iteration_ = iteration
            
            # Generate neighbor solution
            new_solution = neighbor_function(current_solution)
            new_cost = cost_function(new_solution)
            
            total_attempts += 1
            
            # Accept or reject new solution
            if self.should_accept(current_cost, new_cost, temperature):
                current_solution = new_solution
                current_cost = new_cost
                acceptance_count += 1
            
            # Update best solution
            if new_cost < self.best_cost_:
                self.best_solution_ = new_solution
                self.best_cost_ = new_cost
            
            # Adapt parameters periodically
            if iteration % self.adaptation_interval == 0 and total_attempts > 0:
                acceptance_rate = acceptance_count / total_attempts
                self.acceptance_rates_.append(acceptance_rate)
                self.adapt_parameters(iteration, acceptance_rate)
                
                # Reset counters
                acceptance_count = 0
                total_attempts = 0
            
            # Track history
            self.cost_history_.append(current_cost)
            self.temperature_history_.append(temperature)
            
            # Update temperature
            temperature = self.cooling_schedule(iteration, temperature)
            
            # Check stopping criteria
            if temperature < self.final_temp:
                break
        
        return self.best_solution_, self.best_cost_


class ParallelTempering:
    """
    Parallel Tempering (Replica Exchange) variant of Simulated Annealing.
    
    Runs multiple SA chains at different temperatures and exchanges
    configurations between them.
    """
    
    def __init__(self, temperatures: List[float], exchange_interval: int = 10,
                 max_iterations: int = 10000, random_state: Optional[int] = None):
        """
        Initialize Parallel Tempering.
        
        Args:
            temperatures: List of temperatures for different replicas
            exchange_interval: Iterations between replica exchanges
            max_iterations: Maximum iterations
            random_state: Random seed
        """
        self.temperatures = sorted(temperatures, reverse=True)  # Highest first
        self.n_replicas = len(temperatures)
        self.exchange_interval = exchange_interval
        self.max_iterations = max_iterations
        self.random_state = random_state
        
        # Initialize SA instances for each replica
        self.replicas = []
        for temp in self.temperatures:
            sa = SimulatedAnnealing(
                initial_temp=temp,
                final_temp=temp,  # Keep temperature constant
                cooling_rate=1.0,  # No cooling
                max_iterations=max_iterations,
                random_state=random_state
            )
            self.replicas.append(sa)
        
        self.best_solution_: Optional[Any] = None
        self.best_cost_: float = float('inf')
        self.exchange_history_: List[List[int]] = []
        
        if random_state is not None:
            random.seed(random_state)
    
    def exchange_probability(self, cost1: float, cost2: float,
                           temp1: float, temp2: float) -> float:
        """
        Calculate probability of exchanging replicas.
        
        Args:
            cost1: Cost of first replica
            cost2: Cost of second replica
            temp1: Temperature of first replica
            temp2: Temperature of second replica
            
        Returns:
            Exchange probability
        """
        if temp1 == temp2:
            return 0.0
        
        delta = (cost1 - cost2) * (1/temp1 - 1/temp2)
        return min(1.0, math.exp(delta))
    
    def attempt_exchange(self, replica1_idx: int, replica2_idx: int,
                        solutions: List[Any], costs: List[float]) -> bool:
        """
        Attempt to exchange configurations between two replicas.
        
        Args:
            replica1_idx: Index of first replica
            replica2_idx: Index of second replica
            solutions: Current solutions
            costs: Current costs
            
        Returns:
            True if exchange occurred
        """
        temp1 = self.temperatures[replica1_idx]
        temp2 = self.temperatures[replica2_idx]
        cost1 = costs[replica1_idx]
        cost2 = costs[replica2_idx]
        
        prob = self.exchange_probability(cost1, cost2, temp1, temp2)
        
        if random.random() < prob:
            # Exchange solutions
            solutions[replica1_idx], solutions[replica2_idx] = \
                solutions[replica2_idx], solutions[replica1_idx]
            costs[replica1_idx], costs[replica2_idx] = \
                costs[replica2_idx], costs[replica1_idx]
            return True
        
        return False
    
    def optimize(self, cost_function: Callable[[Any], float],
                initial_solution: Any,
                neighbor_function: Callable[[Any], Any]) -> Tuple[Any, float]:
        """
        Run parallel tempering optimization.
        
        Args:
            cost_function: Function to minimize
            initial_solution: Starting solution
            neighbor_function: Function to generate neighbors
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        # Initialize replicas with copies of initial solution
        current_solutions = [initial_solution for _ in range(self.n_replicas)]
        current_costs = [cost_function(sol) for sol in current_solutions]
        
        self.best_solution_ = min(current_solutions, key=cost_function)
        self.best_cost_ = min(current_costs)
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update each replica
            for i in range(self.n_replicas):
                new_solution = neighbor_function(current_solutions[i])
                new_cost = cost_function(new_solution)
                
                # Accept/reject using replica's temperature
                temp = self.temperatures[i]
                if self.replicas[i].should_accept(current_costs[i], new_cost, temp):
                    current_solutions[i] = new_solution
                    current_costs[i] = new_cost
                
                # Update global best
                if new_cost < self.best_cost_:
                    self.best_solution_ = new_solution
                    self.best_cost_ = new_cost
            
            # Attempt replica exchanges
            if iteration % self.exchange_interval == 0:
                exchanges = []
                for i in range(self.n_replicas - 1):
                    if self.attempt_exchange(i, i + 1, current_solutions, current_costs):
                        exchanges.append((i, i + 1))
                self.exchange_history_.append(exchanges)
        
        return self.best_solution_, self.best_cost_
