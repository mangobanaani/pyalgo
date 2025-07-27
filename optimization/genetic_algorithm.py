"""
Genetic Algorithm Implementation

Evolutionary optimization algorithm that evolves a population of candidate
solutions using selection, crossover, and mutation operations.
"""

import random
import math
from typing import List, Tuple, Callable, Optional, Union, Any
from abc import ABC, abstractmethod


class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    def __init__(self, chromosome: List[Any], fitness: Optional[float] = None):
        self.chromosome = chromosome
        self.fitness = fitness
        self.age = 0
    
    def __str__(self) -> str:
        return f"Individual(fitness={self.fitness}, chromosome={self.chromosome})"
    
    def __lt__(self, other: 'Individual') -> bool:
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness


class GeneticAlgorithm:
    """
    Generic Genetic Algorithm implementation.
    
    Evolves a population of candidate solutions through selection,
    crossover, and mutation operations to optimize a fitness function.
    """
    
    def __init__(self, population_size: int = 100, generations: int = 100,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 elite_size: int = 2, tournament_size: int = 3,
                 random_state: Optional[int] = None):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            random_state: Random seed for reproducibility
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        # Evolution tracking
        self.best_individual_: Optional[Individual] = None
        self.best_fitness_history_: List[float] = []
        self.average_fitness_history_: List[float] = []
        self.population_: List[Individual] = []
        self.generation_: int = 0
        
        if random_state is not None:
            random.seed(random_state)
    
    def create_individual(self, chromosome_length: int) -> Individual:
        """
        Create a random individual.
        
        Args:
            chromosome_length: Length of chromosome
            
        Returns:
            Random individual (to be overridden by subclasses)
        """
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        return Individual(chromosome)
    
    def evaluate_fitness(self, individual: Individual, 
                        fitness_function: Callable[[List[Any]], float]) -> float:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            fitness_function: Function to calculate fitness
            
        Returns:
            Fitness value
        """
        return fitness_function(individual.chromosome)
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Select individual using tournament selection.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected individual
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness or 0)
    
    def roulette_wheel_selection(self, population: List[Individual]) -> Individual:
        """
        Select individual using roulette wheel selection.
        
        Args:
            population: Population to select from
            
        Returns:
            Selected individual
        """
        # Handle negative fitness values by shifting
        fitnesses = [ind.fitness or 0 for ind in population]
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population)
        
        # Spin the wheel
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(fitnesses):
            current += fitness
            if current >= pick:
                return population[i]
        
        return population[-1]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        # Single-point crossover
        length = len(parent1.chromosome)
        crossover_point = random.randint(1, length - 1)
        
        offspring1_chromosome = (parent1.chromosome[:crossover_point] + 
                                parent2.chromosome[crossover_point:])
        offspring2_chromosome = (parent2.chromosome[:crossover_point] + 
                                parent1.chromosome[crossover_point:])
        
        return Individual(offspring1_chromosome), Individual(offspring2_chromosome)
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        mutated_chromosome = individual.chromosome.copy()
        
        for i in range(len(mutated_chromosome)):
            if random.random() < self.mutation_rate:
                # Binary mutation (flip bit)
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        
        return Individual(mutated_chromosome)
    
    def create_initial_population(self, chromosome_length: int) -> List[Individual]:
        """
        Create initial population.
        
        Args:
            chromosome_length: Length of each chromosome
            
        Returns:
            Initial population
        """
        return [self.create_individual(chromosome_length) 
                for _ in range(self.population_size)]
    
    def evaluate_population(self, population: List[Individual],
                          fitness_function: Callable[[List[Any]], float]) -> None:
        """
        Evaluate fitness for entire population.
        
        Args:
            population: Population to evaluate
            fitness_function: Fitness function
        """
        for individual in population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual, fitness_function)
    
    def get_elite(self, population: List[Individual]) -> List[Individual]:
        """
        Get elite individuals from population.
        
        Args:
            population: Population to select from
            
        Returns:
            Elite individuals
        """
        sorted_population = sorted(population, key=lambda x: x.fitness or 0, reverse=True)
        return sorted_population[:self.elite_size]
    
    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """
        Evolve one generation.
        
        Args:
            population: Current population
            
        Returns:
            New population
        """
        new_population = []
        
        # Preserve elite individuals
        elite = self.get_elite(population)
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1 = Individual(parent1.chromosome.copy())
                offspring2 = Individual(parent2.chromosome.copy())
            
            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def optimize(self, fitness_function: Callable[[List[Any]], float],
                chromosome_length: int) -> Tuple[Individual, List[float]]:
        """
        Run genetic algorithm optimization.
        
        Args:
            fitness_function: Function to optimize
            chromosome_length: Length of chromosome
            
        Returns:
            Tuple of (best_individual, fitness_history)
        """
        # Initialize population
        population = self.create_initial_population(chromosome_length)
        self.evaluate_population(population, fitness_function)
        
        # Track best individual
        self.best_individual_ = max(population, key=lambda x: x.fitness or 0)
        self.best_fitness_history_ = []
        self.average_fitness_history_ = []
        
        # Evolution loop
        for generation in range(self.generations):
            self.generation_ = generation
            
            # Track statistics
            fitnesses = [ind.fitness or 0 for ind in population]
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            self.best_fitness_history_.append(best_fitness)
            self.average_fitness_history_.append(avg_fitness)
            
            # Update best individual
            current_best = max(population, key=lambda x: x.fitness or 0)
            if current_best.fitness > self.best_individual_.fitness:
                self.best_individual_ = current_best
            
            # Evolve next generation
            if generation < self.generations - 1:
                population = self.evolve_generation(population)
                self.evaluate_population(population, fitness_function)
        
        self.population_ = population
        return self.best_individual_, self.best_fitness_history_


class BinaryGA(GeneticAlgorithm):
    """Genetic Algorithm for binary optimization problems."""
    
    def create_individual(self, chromosome_length: int) -> Individual:
        """Create random binary individual."""
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        return Individual(chromosome)
    
    def mutate(self, individual: Individual) -> Individual:
        """Binary mutation (bit flip)."""
        mutated_chromosome = individual.chromosome.copy()
        
        for i in range(len(mutated_chromosome)):
            if random.random() < self.mutation_rate:
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
        
        return Individual(mutated_chromosome)


class RealValuedGA(GeneticAlgorithm):
    """Genetic Algorithm for real-valued optimization problems."""
    
    def __init__(self, bounds: List[Tuple[float, float]], **kwargs):
        """
        Initialize real-valued GA.
        
        Args:
            bounds: List of (min, max) bounds for each dimension
            **kwargs: Other GA parameters
        """
        super().__init__(**kwargs)
        self.bounds = bounds
        self.n_dimensions = len(bounds)
    
    def create_individual(self, chromosome_length: int = None) -> Individual:
        """Create random real-valued individual."""
        chromosome = []
        for min_val, max_val in self.bounds:
            value = random.uniform(min_val, max_val)
            chromosome.append(value)
        return Individual(chromosome)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover for real values."""
        alpha = 0.5  # Blending parameter
        
        offspring1_chromosome = []
        offspring2_chromosome = []
        
        for i in range(len(parent1.chromosome)):
            p1_val = parent1.chromosome[i]
            p2_val = parent2.chromosome[i]
            
            # Blend crossover
            min_val, max_val = min(p1_val, p2_val), max(p1_val, p2_val)
            range_val = max_val - min_val
            
            lower_bound = min_val - alpha * range_val
            upper_bound = max_val + alpha * range_val
            
            # Respect original bounds
            bound_min, bound_max = self.bounds[i]
            lower_bound = max(lower_bound, bound_min)
            upper_bound = min(upper_bound, bound_max)
            
            offspring1_chromosome.append(random.uniform(lower_bound, upper_bound))
            offspring2_chromosome.append(random.uniform(lower_bound, upper_bound))
        
        return Individual(offspring1_chromosome), Individual(offspring2_chromosome)
    
    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation for real values."""
        mutated_chromosome = individual.chromosome.copy()
        
        for i in range(len(mutated_chromosome)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                min_val, max_val = self.bounds[i]
                mutation_strength = (max_val - min_val) * 0.1
                
                new_value = mutated_chromosome[i] + random.gauss(0, mutation_strength)
                new_value = max(min_val, min(max_val, new_value))  # Clip to bounds
                mutated_chromosome[i] = new_value
        
        return Individual(mutated_chromosome)
    
    def optimize_function(self, fitness_function: Callable[[List[float]], float]) -> Tuple[Individual, List[float]]:
        """
        Optimize real-valued function.
        
        Args:
            fitness_function: Function to optimize
            
        Returns:
            Tuple of (best_individual, fitness_history)
        """
        return self.optimize(fitness_function, self.n_dimensions)


class MultiObjectiveGA(GeneticAlgorithm):
    """
    Multi-objective Genetic Algorithm using NSGA-II approach.
    
    Optimizes multiple objectives simultaneously using Pareto dominance.
    """
    
    def __init__(self, n_objectives: int = 2, **kwargs):
        """
        Initialize multi-objective GA.
        
        Args:
            n_objectives: Number of objectives
            **kwargs: Other GA parameters
        """
        super().__init__(**kwargs)
        self.n_objectives = n_objectives
        self.pareto_front_: List[Individual] = []
    
    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Check if individual 1 dominates individual 2.
        
        Args:
            ind1: First individual
            ind2: Second individual
            
        Returns:
            True if ind1 dominates ind2
        """
        if not hasattr(ind1, 'objectives') or not hasattr(ind2, 'objectives'):
            return False
        
        better_in_any = False
        for i in range(self.n_objectives):
            if ind1.objectives[i] < ind2.objectives[i]:
                return False
            elif ind1.objectives[i] > ind2.objectives[i]:
                better_in_any = True
        
        return better_in_any
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting algorithm.
        
        Args:
            population: Population to sort
            
        Returns:
            List of fronts (lists of individuals)
        """
        fronts = [[]]
        
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            
            for other in population:
                if self.dominates(individual, other):
                    individual.dominated_solutions.append(other)
                elif self.dominates(other, individual):
                    individual.domination_count += 1
            
            if individual.domination_count == 0:
                fronts[0].append(individual)
        
        i = 0
        while fronts[i]:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        next_front.append(dominated)
            
            if next_front:
                fronts.append(next_front)
            i += 1
        
        return fronts[:-1]  # Remove empty last front
    
    def calculate_crowding_distance(self, front: List[Individual]) -> None:
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: Front to calculate distances for
        """
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Initialize distances
        for individual in front:
            individual.crowding_distance = 0
        
        for obj_idx in range(self.n_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Set boundary points to infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distances for intermediate points
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i+1].objectives[obj_idx] - 
                              front[i-1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance
