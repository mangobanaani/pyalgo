"""
Particle Swarm Optimization Implementation

Implements Particle Swarm Optimization algorithm for global optimization
of continuous functions with various variants and improvements.
"""

import math
import random
from typing import List, Tuple, Callable, Dict, Any, Optional


class Particle:
    """Individual particle in the swarm."""
    
    def __init__(self, dimensions: int, bounds: List[Tuple[float, float]]):
        """
        Initialize particle with random position and velocity.
        
        Args:
            dimensions: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
        """
        self.dimensions = dimensions
        self.bounds = bounds
        
        # Initialize position randomly within bounds
        self.position = []
        for i in range(dimensions):
            min_val, max_val = bounds[i]
            self.position.append(random.uniform(min_val, max_val))
        
        # Initialize velocity randomly
        self.velocity = []
        for i in range(dimensions):
            min_val, max_val = bounds[i]
            range_val = max_val - min_val
            self.velocity.append(random.uniform(-range_val * 0.1, range_val * 0.1))
        
        # Personal best
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')
        
        # Current fitness
        self.fitness = float('inf')
    
    def evaluate(self, objective_function: Callable[[List[float]], float]):
        """
        Evaluate particle's fitness.
        
        Args:
            objective_function: Function to minimize
        """
        self.fitness = objective_function(self.position)
        
        # Update personal best
        if self.fitness < self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position.copy()
    
    def update_velocity(self, global_best_position: List[float],
                       w: float, c1: float, c2: float):
        """
        Update particle velocity.
        
        Args:
            global_best_position: Global best position
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        for i in range(self.dimensions):
            r1 = random.random()
            r2 = random.random()
            
            # Velocity update equation
            cognitive = c1 * r1 * (self.personal_best_position[i] - self.position[i])
            social = c2 * r2 * (global_best_position[i] - self.position[i])
            
            self.velocity[i] = w * self.velocity[i] + cognitive + social
            
            # Velocity clamping
            min_val, max_val = self.bounds[i]
            max_velocity = (max_val - min_val) * 0.2
            self.velocity[i] = max(-max_velocity, min(max_velocity, self.velocity[i]))
    
    def update_position(self):
        """Update particle position based on velocity."""
        for i in range(self.dimensions):
            self.position[i] += self.velocity[i]
            
            # Boundary handling
            min_val, max_val = self.bounds[i]
            if self.position[i] < min_val:
                self.position[i] = min_val
                self.velocity[i] = 0  # Stop at boundary
            elif self.position[i] > max_val:
                self.position[i] = max_val
                self.velocity[i] = 0  # Stop at boundary


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization Algorithm.
    
    Implements PSO for global optimization of continuous functions
    with various parameter strategies and improvements.
    """
    
    def __init__(self, n_particles: int = 30, max_iterations: int = 1000,
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0,
                 w_strategy: str = 'constant', tolerance: float = 1e-6,
                 random_state: int = None):
        """
        Initialize PSO optimizer.
        
        Args:
            n_particles: Number of particles in swarm
            max_iterations: Maximum number of iterations
            w: Inertia weight (or initial weight for adaptive strategies)
            c1: Cognitive coefficient (personal best influence)
            c2: Social coefficient (global best influence)
            w_strategy: Inertia weight strategy ('constant', 'linear', 'adaptive')
            tolerance: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w_init = w
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_strategy = w_strategy
        self.tolerance = tolerance
        
        # Optimization results
        self.best_position = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.convergence_iteration = None
        
        # Swarm
        self.swarm = []
        
        if random_state is not None:
            random.seed(random_state)
    
    def optimize(self, objective_function: Callable[[List[float]], float],
                bounds: List[Tuple[float, float]], verbose: bool = False) -> Dict[str, Any]:
        """
        Optimize the objective function.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each dimension
            verbose: Print optimization progress
            
        Returns:
            Optimization results dictionary
        """
        dimensions = len(bounds)
        
        # Initialize swarm
        self.swarm = []
        for _ in range(self.n_particles):
            particle = Particle(dimensions, bounds)
            particle.evaluate(objective_function)
            self.swarm.append(particle)
        
        # Find initial global best
        self._update_global_best()
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            # Update inertia weight
            self._update_inertia_weight(iteration)
            
            # Update particles
            for particle in self.swarm:
                particle.update_velocity(self.best_position, self.w, self.c1, self.c2)
                particle.update_position()
                particle.evaluate(objective_function)
            
            # Update global best
            previous_best = self.best_fitness
            self._update_global_best()
            
            # Store fitness history
            self.fitness_history.append(self.best_fitness)
            
            # Check convergence
            if abs(previous_best - self.best_fitness) < self.tolerance:
                if self.convergence_iteration is None:
                    self.convergence_iteration = iteration
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness:.6f}")
        
        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'n_iterations': self.max_iterations,
            'convergence_iteration': self.convergence_iteration,
            'fitness_history': self.fitness_history
        }
    
    def _update_global_best(self):
        """Update global best position and fitness."""
        for particle in self.swarm:
            if particle.fitness < self.best_fitness:
                self.best_fitness = particle.fitness
                self.best_position = particle.position.copy()
    
    def _update_inertia_weight(self, iteration: int):
        """
        Update inertia weight based on strategy.
        
        Args:
            iteration: Current iteration number
        """
        if self.w_strategy == 'constant':
            # Keep constant weight
            pass
        
        elif self.w_strategy == 'linear':
            # Linear decrease from w_init to 0.1
            w_min = 0.1
            self.w = self.w_init - (self.w_init - w_min) * iteration / self.max_iterations
        
        elif self.w_strategy == 'adaptive':
            # Adaptive weight based on swarm diversity
            diversity = self._calculate_diversity()
            if diversity > 0.1:
                self.w = 0.9  # High diversity, high exploration
            else:
                self.w = 0.4  # Low diversity, high exploitation
        
        else:
            raise ValueError(f"Unknown inertia weight strategy: {self.w_strategy}")
    
    def _calculate_diversity(self) -> float:
        """
        Calculate swarm diversity.
        
        Returns:
            Normalized diversity measure
        """
        if len(self.swarm) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.swarm)):
            for j in range(i + 1, len(self.swarm)):
                distance = sum((self.swarm[i].position[k] - self.swarm[j].position[k]) ** 2
                             for k in range(len(self.swarm[i].position)))
                total_distance += math.sqrt(distance)
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_distance / count


class AdvancedPSO(ParticleSwarmOptimizer):
    """
    Advanced Particle Swarm Optimization with additional features.
    
    Includes constriction factor, neighborhood topologies, and
    adaptive parameter control.
    """
    
    def __init__(self, n_particles: int = 30, max_iterations: int = 1000,
                 phi1: float = 2.05, phi2: float = 2.05,
                 topology: str = 'global', neighborhood_size: int = 3,
                 mutation_rate: float = 0.05, random_state: int = None):
        """
        Initialize Advanced PSO.
        
        Args:
            n_particles: Number of particles
            max_iterations: Maximum iterations
            phi1: Acceleration coefficient 1
            phi2: Acceleration coefficient 2
            topology: Swarm topology ('global', 'ring', 'star')
            neighborhood_size: Size of neighborhood for local topologies
            mutation_rate: Probability of particle mutation
            random_state: Random seed
        """
        # Calculate constriction factor
        phi = phi1 + phi2
        if phi > 4:
            chi = 2 / abs(2 - phi - math.sqrt(phi ** 2 - 4 * phi))
        else:
            chi = 1.0
        
        # Initialize with constriction parameters
        super().__init__(
            n_particles=n_particles,
            max_iterations=max_iterations,
            w=chi,
            c1=chi * phi1,
            c2=chi * phi2,
            w_strategy='constant',
            random_state=random_state
        )
        
        self.topology = topology
        self.neighborhood_size = neighborhood_size
        self.mutation_rate = mutation_rate
        self.neighborhoods = []
    
    def optimize(self, objective_function: Callable[[List[float]], float],
                bounds: List[Tuple[float, float]], verbose: bool = False) -> Dict[str, Any]:
        """
        Optimize using advanced PSO features.
        
        Args:
            objective_function: Function to minimize
            bounds: Bounds for each dimension
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        dimensions = len(bounds)
        
        # Initialize swarm
        self.swarm = []
        for _ in range(self.n_particles):
            particle = Particle(dimensions, bounds)
            particle.evaluate(objective_function)
            self.swarm.append(particle)
        
        # Set up topology
        self._setup_topology()
        
        # Find initial global best
        self._update_global_best()
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            # Update particles with topology-based best
            for i, particle in enumerate(self.swarm):
                local_best = self._get_local_best(i)
                particle.update_velocity(local_best, self.w, self.c1, self.c2)
                particle.update_position()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    self._mutate_particle(particle, bounds)
                
                particle.evaluate(objective_function)
            
            # Update global best
            previous_best = self.best_fitness
            self._update_global_best()
            
            # Store fitness history
            self.fitness_history.append(self.best_fitness)
            
            # Check convergence
            if abs(previous_best - self.best_fitness) < self.tolerance:
                if self.convergence_iteration is None:
                    self.convergence_iteration = iteration
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness:.6f}")
        
        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'n_iterations': self.max_iterations,
            'convergence_iteration': self.convergence_iteration,
            'fitness_history': self.fitness_history
        }
    
    def _setup_topology(self):
        """Setup neighborhood topology."""
        if self.topology == 'global':
            # All particles connected to all others
            self.neighborhoods = [list(range(self.n_particles)) for _ in range(self.n_particles)]
        
        elif self.topology == 'ring':
            # Ring topology: each particle connected to nearest neighbors
            self.neighborhoods = []
            for i in range(self.n_particles):
                neighborhood = []
                for j in range(-self.neighborhood_size // 2, self.neighborhood_size // 2 + 1):
                    neighbor = (i + j) % self.n_particles
                    neighborhood.append(neighbor)
                self.neighborhoods.append(neighborhood)
        
        elif self.topology == 'star':
            # Star topology: all particles connected to best particle
            # Will be updated dynamically
            self.neighborhoods = [[0] * self.n_particles for _ in range(self.n_particles)]
    
    def _get_local_best(self, particle_index: int) -> List[float]:
        """
        Get local best position for a particle based on topology.
        
        Args:
            particle_index: Index of the particle
            
        Returns:
            Local best position
        """
        if self.topology == 'global':
            return self.best_position
        
        elif self.topology == 'ring':
            neighborhood = self.neighborhoods[particle_index]
            best_fitness = float('inf')
            best_position = None
            
            for neighbor_idx in neighborhood:
                if neighbor_idx < len(self.swarm):
                    neighbor = self.swarm[neighbor_idx]
                    if neighbor.personal_best_fitness < best_fitness:
                        best_fitness = neighbor.personal_best_fitness
                        best_position = neighbor.personal_best_position
            
            return best_position if best_position is not None else self.best_position
        
        elif self.topology == 'star':
            # In star topology, return global best
            return self.best_position
    
    def _mutate_particle(self, particle: Particle, bounds: List[Tuple[float, float]]):
        """
        Mutate particle position slightly.
        
        Args:
            particle: Particle to mutate
            bounds: Bounds for each dimension
        """
        for i in range(particle.dimensions):
            if random.random() < 0.1:  # 10% chance to mutate each dimension
                min_val, max_val = bounds[i]
                range_val = max_val - min_val
                mutation = random.gauss(0, range_val * 0.01)  # Small Gaussian mutation
                particle.position[i] += mutation
                
                # Ensure bounds
                particle.position[i] = max(min_val, min(max_val, particle.position[i]))


# Test functions for optimization
def sphere_function(x: List[float]) -> float:
    """Sphere function: f(x) = sum(x_i^2)"""
    return sum(xi ** 2 for xi in x)


def rosenbrock_function(x: List[float]) -> float:
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    result = 0.0
    for i in range(len(x) - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return result


def rastrigin_function(x: List[float]) -> float:
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    A = 10
    n = len(x)
    return A * n + sum(xi ** 2 - A * math.cos(2 * math.pi * xi) for xi in x)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Particle Swarm Optimization...")
    
    # Test on sphere function
    print("\n1. Sphere function optimization:")
    bounds_sphere = [(-5.0, 5.0)] * 2  # 2D sphere
    
    pso = ParticleSwarmOptimizer(n_particles=20, max_iterations=500, random_state=42)
    result = pso.optimize(sphere_function, bounds_sphere, verbose=True)
    
    print(f"Best position: {[f'{x:.6f}' for x in result['best_position']]}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Converged at iteration: {result['convergence_iteration']}")
    
    # Test on Rosenbrock function
    print("\n2. Rosenbrock function optimization:")
    bounds_rosenbrock = [(-2.0, 2.0)] * 2
    
    pso_adaptive = ParticleSwarmOptimizer(
        n_particles=30,
        max_iterations=1000,
        w_strategy='linear',
        random_state=42
    )
    result = pso_adaptive.optimize(rosenbrock_function, bounds_rosenbrock, verbose=False)
    
    print(f"Best position: {[f'{x:.6f}' for x in result['best_position']]}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print("Expected optimum: [1.0, 1.0] with fitness 0.0")
    
    # Test Advanced PSO on Rastrigin function
    print("\n3. Advanced PSO on Rastrigin function:")
    bounds_rastrigin = [(-5.12, 5.12)] * 3  # 3D Rastrigin
    
    advanced_pso = AdvancedPSO(
        n_particles=40,
        max_iterations=800,
        topology='ring',
        neighborhood_size=3,
        mutation_rate=0.1,
        random_state=42
    )
    result = advanced_pso.optimize(rastrigin_function, bounds_rastrigin, verbose=False)
    
    print(f"Best position: {[f'{x:.6f}' for x in result['best_position']]}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print("Expected optimum: [0.0, 0.0, 0.0] with fitness 0.0")
    
    # Performance comparison
    print("\n4. Comparing PSO variants on sphere function:")
    bounds = [(-10.0, 10.0)] * 5  # 5D sphere
    
    variants = [
        ("Standard PSO", ParticleSwarmOptimizer(random_state=42)),
        ("Linear Weight", ParticleSwarmOptimizer(w_strategy='linear', random_state=42)),
        ("Adaptive Weight", ParticleSwarmOptimizer(w_strategy='adaptive', random_state=42)),
        ("Advanced PSO", AdvancedPSO(topology='ring', random_state=42))
    ]
    
    for name, optimizer in variants:
        result = optimizer.optimize(sphere_function, bounds, verbose=False)
        print(f"{name}: {result['best_fitness']:.6f} "
              f"(converged at {result['convergence_iteration']})")
    
    print("\nPSO optimization tests completed!")
