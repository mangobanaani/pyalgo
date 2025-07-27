"""
Differential Evolution Implementation

Implements Differential Evolution algorithm for global optimization
with various strategies and adaptive parameters.
"""

import random
import math
from typing import List, Tuple, Callable, Dict, Any, Optional


class DifferentialEvolution:
    """
    Differential Evolution Algorithm.
    
    Implements DE for global optimization of continuous functions
    with multiple mutation and crossover strategies.
    """
    
    def __init__(self, population_size: int = 50, max_generations: int = 1000,
                 F: float = 0.5, CR: float = 0.9, strategy: str = 'DE/rand/1',
                 tolerance: float = 1e-6, adaptive: bool = False,
                 random_state: int = None):
        """
        Initialize Differential Evolution optimizer.
        
        Args:
            population_size: Size of the population
            max_generations: Maximum number of generations
            F: Differential weight (mutation factor)
            CR: Crossover probability
            strategy: DE strategy ('DE/rand/1', 'DE/best/1', 'DE/rand/2', etc.)
            tolerance: Convergence tolerance
            adaptive: Use adaptive parameter control
            random_state: Random seed for reproducibility
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.tolerance = tolerance
        self.adaptive = adaptive
        
        # Optimization results
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.convergence_generation = None
        
        # Population
        self.population = []
        self.fitness_values = []
        
        # Adaptive parameters
        self.F_history = []
        self.CR_history = []
        
        if random_state is not None:
            random.seed(random_state)
    
    def optimize(self, objective_function: Callable[[List[float]], float],
                bounds: List[Tuple[float, float]], verbose: bool = False) -> Dict[str, Any]:
        """
        Optimize the objective function using Differential Evolution.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            verbose: Print optimization progress
            
        Returns:
            Optimization results dictionary
        """
        dimensions = len(bounds)
        
        # Initialize population
        self._initialize_population(dimensions, bounds)
        
        # Evaluate initial population
        self.fitness_values = []
        for individual in self.population:
            fitness = objective_function(individual)
            self.fitness_values.append(fitness)
        
        # Find initial best
        self._update_best()
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Adaptive parameter control
            if self.adaptive:
                self._adapt_parameters(generation)
            
            # Create new generation
            new_population = []
            new_fitness_values = []
            
            for i in range(self.population_size):
                # Mutation
                mutant = self._mutate(i, self.population)
                
                # Crossover
                trial = self._crossover(self.population[i], mutant, dimensions)
                
                # Boundary handling
                trial = self._handle_boundaries(trial, bounds)
                
                # Selection
                trial_fitness = objective_function(trial)
                
                if trial_fitness <= self.fitness_values[i]:
                    new_population.append(trial)
                    new_fitness_values.append(trial_fitness)
                else:
                    new_population.append(self.population[i])
                    new_fitness_values.append(self.fitness_values[i])
            
            # Update population
            self.population = new_population
            self.fitness_values = new_fitness_values
            
            # Update best solution
            previous_best = self.best_fitness
            self._update_best()
            
            # Store fitness history
            self.fitness_history.append(self.best_fitness)
            
            # Check convergence
            if abs(previous_best - self.best_fitness) < self.tolerance:
                if self.convergence_generation is None:
                    self.convergence_generation = generation
            
            if verbose and generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.6f}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'n_generations': self.max_generations,
            'convergence_generation': self.convergence_generation,
            'fitness_history': self.fitness_history,
            'final_population': self.population,
            'final_fitness_values': self.fitness_values
        }
    
    def _initialize_population(self, dimensions: int, bounds: List[Tuple[float, float]]):
        """
        Initialize population randomly within bounds.
        
        Args:
            dimensions: Number of dimensions
            bounds: Bounds for each dimension
        """
        self.population = []
        for _ in range(self.population_size):
            individual = []
            for i in range(dimensions):
                min_val, max_val = bounds[i]
                individual.append(random.uniform(min_val, max_val))
            self.population.append(individual)
    
    def _mutate(self, target_index: int, population: List[List[float]]) -> List[float]:
        """
        Create mutant vector using specified strategy.
        
        Args:
            target_index: Index of target vector
            population: Current population
            
        Returns:
            Mutant vector
        """
        if self.strategy == 'DE/rand/1':
            return self._mutate_rand_1(target_index, population)
        elif self.strategy == 'DE/best/1':
            return self._mutate_best_1(target_index, population)
        elif self.strategy == 'DE/rand/2':
            return self._mutate_rand_2(target_index, population)
        elif self.strategy == 'DE/best/2':
            return self._mutate_best_2(target_index, population)
        elif self.strategy == 'DE/current-to-best/1':
            return self._mutate_current_to_best_1(target_index, population)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _mutate_rand_1(self, target_index: int, population: List[List[float]]) -> List[float]:
        """DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)"""
        indices = self._select_random_indices(target_index, 3)
        r1, r2, r3 = indices
        
        mutant = []
        for i in range(len(population[0])):
            mutant.append(population[r1][i] + self.F * (population[r2][i] - population[r3][i]))
        
        return mutant
    
    def _mutate_best_1(self, target_index: int, population: List[List[float]]) -> List[float]:
        """DE/best/1: v = x_best + F * (x_r1 - x_r2)"""
        indices = self._select_random_indices(target_index, 2)
        r1, r2 = indices
        
        best_index = self.fitness_values.index(min(self.fitness_values))
        
        mutant = []
        for i in range(len(population[0])):
            mutant.append(population[best_index][i] + self.F * (population[r1][i] - population[r2][i]))
        
        return mutant
    
    def _mutate_rand_2(self, target_index: int, population: List[List[float]]) -> List[float]:
        """DE/rand/2: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)"""
        indices = self._select_random_indices(target_index, 5)
        r1, r2, r3, r4, r5 = indices
        
        mutant = []
        for i in range(len(population[0])):
            mutant.append(population[r1][i] + 
                         self.F * (population[r2][i] - population[r3][i]) +
                         self.F * (population[r4][i] - population[r5][i]))
        
        return mutant
    
    def _mutate_best_2(self, target_index: int, population: List[List[float]]) -> List[float]:
        """DE/best/2: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)"""
        indices = self._select_random_indices(target_index, 4)
        r1, r2, r3, r4 = indices
        
        best_index = self.fitness_values.index(min(self.fitness_values))
        
        mutant = []
        for i in range(len(population[0])):
            mutant.append(population[best_index][i] + 
                         self.F * (population[r1][i] - population[r2][i]) +
                         self.F * (population[r3][i] - population[r4][i]))
        
        return mutant
    
    def _mutate_current_to_best_1(self, target_index: int, population: List[List[float]]) -> List[float]:
        """DE/current-to-best/1: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)"""
        indices = self._select_random_indices(target_index, 2)
        r1, r2 = indices
        
        best_index = self.fitness_values.index(min(self.fitness_values))
        
        mutant = []
        for i in range(len(population[0])):
            mutant.append(population[target_index][i] + 
                         self.F * (population[best_index][i] - population[target_index][i]) +
                         self.F * (population[r1][i] - population[r2][i]))
        
        return mutant
    
    def _select_random_indices(self, target_index: int, count: int) -> List[int]:
        """
        Select random indices different from target index.
        
        Args:
            target_index: Index to exclude
            count: Number of indices to select
            
        Returns:
            List of random indices
        """
        available_indices = [i for i in range(self.population_size) if i != target_index]
        return random.sample(available_indices, count)
    
    def _crossover(self, target: List[float], mutant: List[float], dimensions: int) -> List[float]:
        """
        Perform crossover between target and mutant vectors.
        
        Args:
            target: Target vector
            mutant: Mutant vector
            dimensions: Number of dimensions
            
        Returns:
            Trial vector
        """
        trial = []
        j_rand = random.randint(0, dimensions - 1)  # Ensure at least one component from mutant
        
        for j in range(dimensions):
            if random.random() <= self.CR or j == j_rand:
                trial.append(mutant[j])
            else:
                trial.append(target[j])
        
        return trial
    
    def _handle_boundaries(self, individual: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
        """
        Handle boundary constraints.
        
        Args:
            individual: Individual to check
            bounds: Bounds for each dimension
            
        Returns:
            Individual with corrected boundaries
        """
        corrected = []
        for i, value in enumerate(individual):
            min_val, max_val = bounds[i]
            
            # Clipping strategy
            corrected_value = max(min_val, min(max_val, value))
            corrected.append(corrected_value)
        
        return corrected
    
    def _update_best(self):
        """Update best solution and fitness."""
        min_fitness_index = self.fitness_values.index(min(self.fitness_values))
        
        if self.fitness_values[min_fitness_index] < self.best_fitness:
            self.best_fitness = self.fitness_values[min_fitness_index]
            self.best_solution = self.population[min_fitness_index].copy()
    
    def _adapt_parameters(self, generation: int):
        """
        Adapt F and CR parameters based on success rates.
        
        Args:
            generation: Current generation number
        """
        # Simple adaptive strategy
        if generation > 0 and generation % 50 == 0:
            # Increase F if no improvement in recent generations
            recent_improvements = []
            for i in range(max(0, len(self.fitness_history) - 50), len(self.fitness_history)):
                if i > 0:
                    improvement = self.fitness_history[i-1] - self.fitness_history[i]
                    recent_improvements.append(improvement)
            
            if recent_improvements:
                avg_improvement = sum(recent_improvements) / len(recent_improvements)
                
                if avg_improvement < self.tolerance:
                    # No significant improvement, increase exploration
                    self.F = min(1.0, self.F * 1.1)
                    self.CR = max(0.1, self.CR * 0.9)
                else:
                    # Good improvement, increase exploitation
                    self.F = max(0.1, self.F * 0.9)
                    self.CR = min(1.0, self.CR * 1.1)
        
        # Store parameter history
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)


class SelfAdaptiveDifferentialEvolution(DifferentialEvolution):
    """
    Self-Adaptive Differential Evolution (SaDE).
    
    Implements DE with self-adaptive control parameters where
    F and CR values evolve along with the population.
    """
    
    def __init__(self, population_size: int = 50, max_generations: int = 1000,
                 tolerance: float = 1e-6, random_state: int = None):
        """
        Initialize Self-Adaptive DE.
        
        Args:
            population_size: Size of population
            max_generations: Maximum generations
            tolerance: Convergence tolerance
            random_state: Random seed
        """
        super().__init__(
            population_size=population_size,
            max_generations=max_generations,
            F=0.5,  # Will be adapted
            CR=0.9,  # Will be adapted
            strategy='DE/rand/1',
            tolerance=tolerance,
            adaptive=False,  # We handle adaptation ourselves
            random_state=random_state
        )
        
        # Self-adaptive parameters
        self.F_values = []
        self.CR_values = []
    
    def optimize(self, objective_function: Callable[[List[float]], float],
                bounds: List[Tuple[float, float]], verbose: bool = False) -> Dict[str, Any]:
        """
        Optimize using self-adaptive DE.
        
        Args:
            objective_function: Function to minimize
            bounds: Bounds for each dimension
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        dimensions = len(bounds)
        
        # Initialize population
        self._initialize_population(dimensions, bounds)
        
        # Initialize control parameters
        self.F_values = [random.uniform(0.1, 1.0) for _ in range(self.population_size)]
        self.CR_values = [random.uniform(0.0, 1.0) for _ in range(self.population_size)]
        
        # Evaluate initial population
        self.fitness_values = []
        for individual in self.population:
            fitness = objective_function(individual)
            self.fitness_values.append(fitness)
        
        # Find initial best
        self._update_best()
        
        # Evolution loop
        for generation in range(self.max_generations):
            new_population = []
            new_fitness_values = []
            new_F_values = []
            new_CR_values = []
            
            for i in range(self.population_size):
                # Use individual's control parameters
                self.F = self.F_values[i]
                self.CR = self.CR_values[i]
                
                # Mutation
                mutant = self._mutate(i, self.population)
                
                # Crossover
                trial = self._crossover(self.population[i], mutant, dimensions)
                
                # Boundary handling
                trial = self._handle_boundaries(trial, bounds)
                
                # Selection
                trial_fitness = objective_function(trial)
                
                if trial_fitness <= self.fitness_values[i]:
                    # Trial is better, inherit or mutate control parameters
                    new_population.append(trial)
                    new_fitness_values.append(trial_fitness)
                    
                    # Inherit successful parameters with small mutation
                    new_F = self.F_values[i] + random.gauss(0, 0.1)
                    new_F = max(0.1, min(1.0, new_F))
                    new_F_values.append(new_F)
                    
                    new_CR = self.CR_values[i] + random.gauss(0, 0.1)
                    new_CR = max(0.0, min(1.0, new_CR))
                    new_CR_values.append(new_CR)
                else:
                    # Keep old solution and parameters
                    new_population.append(self.population[i])
                    new_fitness_values.append(self.fitness_values[i])
                    new_F_values.append(self.F_values[i])
                    new_CR_values.append(self.CR_values[i])
            
            # Update population and parameters
            self.population = new_population
            self.fitness_values = new_fitness_values
            self.F_values = new_F_values
            self.CR_values = new_CR_values
            
            # Update best solution
            previous_best = self.best_fitness
            self._update_best()
            
            # Store fitness history
            self.fitness_history.append(self.best_fitness)
            
            # Check convergence
            if abs(previous_best - self.best_fitness) < self.tolerance:
                if self.convergence_generation is None:
                    self.convergence_generation = generation
            
            if verbose and generation % 100 == 0:
                avg_F = sum(self.F_values) / len(self.F_values)
                avg_CR = sum(self.CR_values) / len(self.CR_values)
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.6f}, "
                      f"Avg F = {avg_F:.3f}, Avg CR = {avg_CR:.3f}")
        
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'n_generations': self.max_generations,
            'convergence_generation': self.convergence_generation,
            'fitness_history': self.fitness_history,
            'final_F_values': self.F_values,
            'final_CR_values': self.CR_values
        }


# Test functions
def ackley_function(x: List[float]) -> float:
    """Ackley function: multimodal test function"""
    n = len(x)
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    
    return (-20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - 
            math.exp(sum2 / n) + 20 + math.e)


def griewank_function(x: List[float]) -> float:
    """Griewank function: multimodal test function"""
    sum_part = sum(xi ** 2 for xi in x) / 4000
    prod_part = 1.0
    for i, xi in enumerate(x):
        prod_part *= math.cos(xi / math.sqrt(i + 1))
    
    return sum_part - prod_part + 1


# Example usage and testing
if __name__ == "__main__":
    print("Testing Differential Evolution...")
    
    # Test on Ackley function
    print("\n1. Ackley function optimization:")
    bounds_ackley = [(-32.768, 32.768)] * 2  # 2D Ackley
    
    de = DifferentialEvolution(
        population_size=30,
        max_generations=500,
        F=0.5,
        CR=0.9,
        strategy='DE/rand/1',
        random_state=42
    )
    result = de.optimize(ackley_function, bounds_ackley, verbose=True)
    
    print(f"Best solution: {[f'{x:.6f}' for x in result['best_solution']]}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print("Expected optimum: [0.0, 0.0] with fitness 0.0")
    
    # Test different strategies
    print("\n2. Comparing DE strategies on Griewank function:")
    bounds_griewank = [(-600.0, 600.0)] * 3  # 3D Griewank
    
    strategies = ['DE/rand/1', 'DE/best/1', 'DE/rand/2', 'DE/current-to-best/1']
    
    for strategy in strategies:
        de_strategy = DifferentialEvolution(
            population_size=40,
            max_generations=300,
            strategy=strategy,
            random_state=42
        )
        result = de_strategy.optimize(griewank_function, bounds_griewank, verbose=False)
        print(f"{strategy}: {result['best_fitness']:.6f}")
    
    # Test Self-Adaptive DE
    print("\n3. Self-Adaptive Differential Evolution:")
    
    sade = SelfAdaptiveDifferentialEvolution(
        population_size=30,
        max_generations=400,
        random_state=42
    )
    result = sade.optimize(ackley_function, bounds_ackley, verbose=True)
    
    print(f"Best solution: {[f'{x:.6f}' for x in result['best_solution']]}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Final F range: {min(result['final_F_values']):.3f} - {max(result['final_F_values']):.3f}")
    print(f"Final CR range: {min(result['final_CR_values']):.3f} - {max(result['final_CR_values']):.3f}")
    
    # Adaptive DE
    print("\n4. Adaptive Differential Evolution:")
    
    adaptive_de = DifferentialEvolution(
        population_size=30,
        max_generations=400,
        adaptive=True,
        random_state=42
    )
    result = adaptive_de.optimize(griewank_function, bounds_griewank, verbose=False)
    
    print(f"Best solution: {[f'{x:.6f}' for x in result['best_solution']]}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Converged at generation: {result['convergence_generation']}")
    
    print("\nDifferential Evolution tests completed!")
