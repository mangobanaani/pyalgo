"""
Optimization Algorithms

This package implements various optimization algorithms including
evolutionary algorithms, metaheuristics, and local search methods.
"""

from .genetic_algorithm import GeneticAlgorithm, BinaryGA, RealValuedGA
from .simulated_annealing import SimulatedAnnealing, AdaptiveSimulatedAnnealing, ParallelTempering
from .particle_swarm import ParticleSwarmOptimizer, AdvancedPSO
from .differential_evolution import DifferentialEvolution, SelfAdaptiveDifferentialEvolution

__all__ = [
    'GeneticAlgorithm',
    'BinaryGA',
    'RealValuedGA',
    'SimulatedAnnealing',
    'AdaptiveSimulatedAnnealing',
    'ParallelTempering',
    'ParticleSwarmOptimizer',
    'AdvancedPSO',
    'DifferentialEvolution',
    'SelfAdaptiveDifferentialEvolution'
]

# Evolutionary Algorithms
from .genetic_algorithm import GeneticAlgorithm, BinaryGA, RealValuedGA
from .differential_evolution import DifferentialEvolution
from .particle_swarm import ParticleSwarmOptimization
from .evolution_strategies import EvolutionStrategy, CovarianceMatrixAdaptation

# Metaheuristics
from .simulated_annealing import SimulatedAnnealing
from .tabu_search import TabuSearch
from .ant_colony import AntColonyOptimization
from .bee_algorithm import BeeAlgorithm, ArtificialBeeColony

# Local Search
from .hill_climbing import HillClimbing, SteepestAscentHillClimbing
from .random_search import RandomSearch
from .local_beam_search import LocalBeamSearch

# Gradient-Based Methods
from .gradient_descent import GradientDescent, StochasticGradientDescent
from .newton_method import NewtonMethod, QuasiNewtonMethod
from .conjugate_gradient import ConjugateGradient
from .lbfgs import LBFGS

# Multi-Objective Optimization
from .nsga import NSGAII, NSGAIII
from .spea import SPEA2
from .moea import MOEAD

# Constraint Handling
from .penalty_methods import PenaltyMethod, AugmentedLagrangian
from .barrier_methods import InteriorPointMethod
from .lagrange_multipliers import LagrangeMultipliers

# Specialized Optimizers
from .bayesian_optimization import BayesianOptimization
from .hyperparameter_tuning import GridSearch, RandomizedSearch
from .combinatorial_optimization import BranchAndBound, CuttingPlane

__all__ = [
    # Evolutionary Algorithms
    'GeneticAlgorithm', 'BinaryGA', 'RealValuedGA',
    'DifferentialEvolution',
    'ParticleSwarmOptimization',
    'EvolutionStrategy', 'CovarianceMatrixAdaptation',
    
    # Metaheuristics
    'SimulatedAnnealing',
    'TabuSearch',
    'AntColonyOptimization',
    'BeeAlgorithm', 'ArtificialBeeColony',
    
    # Local Search
    'HillClimbing', 'SteepestAscentHillClimbing',
    'RandomSearch',
    'LocalBeamSearch',
    
    # Gradient-Based Methods
    'GradientDescent', 'StochasticGradientDescent',
    'NewtonMethod', 'QuasiNewtonMethod',
    'ConjugateGradient',
    'LBFGS',
    
    # Multi-Objective Optimization
    'NSGAII', 'NSGAIII',
    'SPEA2',
    'MOEAD',
    
    # Constraint Handling
    'PenaltyMethod', 'AugmentedLagrangian',
    'InteriorPointMethod',
    'LagrangeMultipliers',
    
    # Specialized Optimizers
    'BayesianOptimization',
    'GridSearch', 'RandomizedSearch',
    'BranchAndBound', 'CuttingPlane'
]
