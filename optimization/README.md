# Optimization Algorithms

This directory contains implementations of various optimization algorithms for solving continuous and discrete optimization problems. The algorithms are designed to find optimal or near-optimal solutions for complex objective functions.

## Available Algorithms

### Evolutionary Algorithms

**Genetic Algorithm** (`genetic_algorithm.py`)
- Standard genetic algorithm with binary and real-valued representations
- Tournament, roulette wheel, and rank selection methods
- Single-point, two-point, and uniform crossover operators
- Bit-flip and Gaussian mutation operators
- Elitism and diversity preservation mechanisms

**Simulated Annealing** (`simulated_annealing.py`)
- Classic simulated annealing with various cooling schedules
- Adaptive simulated annealing with dynamic parameter adjustment
- Parallel tempering for enhanced global exploration
- Multiple neighborhood generation strategies

**Particle Swarm Optimization** (`particle_swarm.py`)
- Standard PSO with inertia weight and acceleration coefficients
- Advanced PSO with constriction factor and topology variants
- Adaptive parameter control strategies
- Ring, star, and global neighborhood topologies

**Differential Evolution** (`differential_evolution.py`)
- Multiple DE strategies: DE/rand/1, DE/best/1, DE/rand/2, etc.
- Self-adaptive parameter control
- Boundary handling mechanisms
- Population diversity maintenance

## Key Features

### Algorithm Characteristics

- **Global Optimization**: All algorithms are designed for global optimization problems
- **Continuous Variables**: Support for real-valued optimization variables
- **Discrete Variables**: Binary genetic algorithm for discrete problems
- **Multi-modal Functions**: Effective on functions with multiple local optima
- **Constraint Handling**: Basic boundary constraint support

### Implementation Features

- **Pure Python**: No external dependencies required
- **Modular Design**: Easy to extend and customize
- **Performance Tracking**: Fitness history and convergence monitoring
- **Reproducible Results**: Random seed support for consistent results
- **Verbose Output**: Optional progress reporting during optimization

## Usage Examples

### Genetic Algorithm
```python
from optimization import GeneticAlgorithm

def objective_function(x):
    return sum(xi**2 for xi in x)  # Sphere function

bounds = [(-5, 5)] * 10  # 10-dimensional problem
ga = GeneticAlgorithm(population_size=50, generations=200)
result = ga.optimize(objective_function, bounds)
print(f"Best solution: {result['best_individual']}")
```

### Simulated Annealing
```python
from optimization import SimulatedAnnealing

sa = SimulatedAnnealing(
    initial_temperature=100,
    final_temperature=0.01,
    cooling_schedule='exponential'
)
result = sa.optimize(objective_function, bounds)
```

### Particle Swarm Optimization
```python
from optimization import ParticleSwarmOptimizer

pso = ParticleSwarmOptimizer(
    n_particles=30,
    max_iterations=1000,
    w_strategy='linear'
)
result = pso.optimize(objective_function, bounds)
```

### Differential Evolution
```python
from optimization import DifferentialEvolution

de = DifferentialEvolution(
    population_size=40,
    max_generations=500,
    strategy='DE/rand/1',
    adaptive=True
)
result = de.optimize(objective_function, bounds)
```

## Algorithm Selection Guide

### Problem Characteristics

**Continuous Optimization**:
- Particle Swarm Optimization: Good for continuous, differentiable functions
- Differential Evolution: Robust for multimodal continuous functions
- Real-valued Genetic Algorithm: Versatile for various problem types

**Discrete/Binary Optimization**:
- Binary Genetic Algorithm: Specialized for binary variables
- Simulated Annealing: Can handle discrete neighborhoods

**Multimodal Functions**:
- Differential Evolution: Excellent exploration capabilities
- Advanced PSO: Multiple swarm topologies for diversity
- Parallel Tempering: Multiple temperature chains

### Performance Considerations

**Fast Convergence**:
- Particle Swarm Optimization with local topology
- Adaptive Differential Evolution
- Simulated Annealing with aggressive cooling

**Global Exploration**:
- Genetic Algorithm with high mutation rate
- Parallel Tempering Simulated Annealing
- Ring topology PSO

**Parameter Sensitivity**:
- Self-Adaptive Differential Evolution (automatic tuning)
- Adaptive Simulated Annealing (dynamic parameters)
- Advanced PSO with constriction factor

## Test Functions

The implementations include several standard test functions:

- **Sphere Function**: Simple unimodal function
- **Rosenbrock Function**: Classic optimization challenge
- **Rastrigin Function**: Highly multimodal test function
- **Ackley Function**: Multimodal with single global optimum
- **Griewank Function**: Multimodal with many local optima

## Performance Metrics

All algorithms provide:
- Best solution found
- Best fitness value achieved
- Convergence iteration/generation
- Fitness history for analysis
- Algorithm-specific metrics (diversity, parameters, etc.)

## Extensions and Customization

The modular design allows for easy extensions:
- Custom objective functions
- New selection/crossover/mutation operators
- Alternative cooling schedules
- Custom neighborhood topologies
- Hybrid algorithm combinations

Each algorithm can be subclassed to add problem-specific enhancements while maintaining the standard interface.
