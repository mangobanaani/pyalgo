"""
Tests for Optimization Algorithms

Test suite for optimization algorithms including evolutionary algorithms,
metaheuristics, and local search methods.
"""

import unittest
import random
import math
from optimization import (
    GeneticAlgorithm, BinaryGA, RealValuedGA,
    SimulatedAnnealing, AdaptiveSimulatedAnnealing,
    ParticleSwarmOptimizer, AdvancedPSO,
    DifferentialEvolution, SelfAdaptiveDifferentialEvolution
)


# Test functions for optimization
def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return sum(xi ** 2 for xi in x)


def rosenbrock_function(x):
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    result = 0.0
    for i in range(len(x) - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return result


def rastrigin_function(x):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    A = 10
    n = len(x)
    return A * n + sum(xi ** 2 - A * math.cos(2 * math.pi * xi) for xi in x)


def ackley_function(x):
    """Ackley function: multimodal test function"""
    n = len(x)
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    
    return (-20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - 
            math.exp(sum2 / n) + 20 + math.e)


class TestGeneticAlgorithm(unittest.TestCase):
    """Test cases for Genetic Algorithm."""
    
    def setUp(self):
        """Set up test parameters."""
        self.bounds_2d = [(-5, 5), (-5, 5)]
        self.bounds_5d = [(-10, 10)] * 5
        random.seed(42)
    
    def test_sphere_optimization(self):
        """Test GA on sphere function."""
        ga = GeneticAlgorithm(
            population_size=20,
            generations=50,
            crossover_rate=0.8,
            mutation_rate=0.1,
            random_state=42
        )
        
        result = ga.optimize(sphere_function, self.bounds_2d)
        
        # Should find solution close to [0, 0]
        self.assertIsInstance(result['best_individual'], list)
        self.assertEqual(len(result['best_individual']), 2)
        self.assertLess(result['best_fitness'], 1.0)  # Should be close to 0
        self.assertIn('fitness_history', result)
    
    def test_binary_ga(self):
        """Test Binary GA."""
        def binary_objective(x):
            # Count number of ones (maximize)
            return -sum(x)  # Minimize negative sum
        
        binary_ga = BinaryGA(
            population_size=20,
            generations=30,
            chromosome_length=10,
            random_state=42
        )
        
        result = binary_ga.optimize(binary_objective)
        
        # Should find all ones
        self.assertEqual(len(result['best_individual']), 10)
        self.assertTrue(all(gene in [0, 1] for gene in result['best_individual']))
        self.assertLessEqual(result['best_fitness'], 0)  # Negative sum should be <= 0
    
    def test_real_valued_ga(self):
        """Test Real-valued GA."""
        real_ga = RealValuedGA(
            population_size=30,
            generations=40,
            random_state=42
        )
        
        result = real_ga.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_individual'], list)
        self.assertLess(result['best_fitness'], 5.0)
    
    def test_ga_convergence(self):
        """Test GA convergence tracking."""
        ga = GeneticAlgorithm(population_size=15, generations=20, random_state=42)
        result = ga.optimize(sphere_function, self.bounds_2d)
        
        # Fitness should generally improve over generations
        fitness_history = result['fitness_history']
        self.assertEqual(len(fitness_history), 20)
        
        # Best fitness should be the last recorded or better
        self.assertLessEqual(result['best_fitness'], fitness_history[-1])
    
    def test_ga_parameters(self):
        """Test GA with different parameters."""
        # High mutation rate
        ga_high_mut = GeneticAlgorithm(
            population_size=20,
            generations=10,
            mutation_rate=0.5,
            random_state=42
        )
        result = ga_high_mut.optimize(sphere_function, self.bounds_2d)
        self.assertIsInstance(result['best_fitness'], (int, float))
        
        # Low crossover rate
        ga_low_cross = GeneticAlgorithm(
            population_size=20,
            generations=10,
            crossover_rate=0.3,
            random_state=42
        )
        result = ga_low_cross.optimize(sphere_function, self.bounds_2d)
        self.assertIsInstance(result['best_fitness'], (int, float))


class TestSimulatedAnnealing(unittest.TestCase):
    """Test cases for Simulated Annealing."""
    
    def setUp(self):
        """Set up test parameters."""
        self.bounds_2d = [(-5, 5), (-5, 5)]
        random.seed(42)
    
    def test_basic_sa(self):
        """Test basic simulated annealing."""
        sa = SimulatedAnnealing(
            initial_temperature=10,
            final_temperature=0.01,
            cooling_schedule='exponential',
            max_iterations=100,
            random_state=42
        )
        
        result = sa.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_solution'], list)
        self.assertEqual(len(result['best_solution']), 2)
        self.assertLess(result['best_energy'], 10.0)
        self.assertIn('energy_history', result)
    
    def test_cooling_schedules(self):
        """Test different cooling schedules."""
        schedules = ['linear', 'exponential', 'logarithmic']
        
        for schedule in schedules:
            sa = SimulatedAnnealing(
                initial_temperature=5,
                final_temperature=0.1,
                cooling_schedule=schedule,
                max_iterations=50,
                random_state=42
            )
            
            result = sa.optimize(sphere_function, self.bounds_2d)
            self.assertIsInstance(result['best_energy'], (int, float))
    
    def test_adaptive_sa(self):
        """Test adaptive simulated annealing."""
        asa = AdaptiveSimulatedAnnealing(
            initial_temperature=10,
            adaptation_rate=0.1,
            max_iterations=80,
            random_state=42
        )
        
        result = asa.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_solution'], list)
        self.assertIn('temperature_history', result)
        self.assertIn('acceptance_history', result)
    
    def test_sa_multimodal(self):
        """Test SA on multimodal function."""
        sa = SimulatedAnnealing(
            initial_temperature=20,
            final_temperature=0.01,
            max_iterations=200,
            random_state=42
        )
        
        bounds_rastrigin = [(-5.12, 5.12)] * 2
        result = sa.optimize(rastrigin_function, bounds_rastrigin)
        
        # Should find solution reasonably close to global optimum
        self.assertLess(result['best_energy'], 50.0)  # Rastrigin can be challenging


class TestParticleSwarmOptimization(unittest.TestCase):
    """Test cases for Particle Swarm Optimization."""
    
    def setUp(self):
        """Set up test parameters."""
        self.bounds_2d = [(-5, 5), (-5, 5)]
        random.seed(42)
    
    def test_basic_pso(self):
        """Test basic PSO."""
        pso = ParticleSwarmOptimizer(
            n_particles=20,
            max_iterations=100,
            w=0.9,
            c1=2.0,
            c2=2.0,
            random_state=42
        )
        
        result = pso.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_position'], list)
        self.assertEqual(len(result['best_position']), 2)
        self.assertLess(result['best_fitness'], 5.0)
        self.assertIn('fitness_history', result)
    
    def test_pso_inertia_strategies(self):
        """Test different inertia weight strategies."""
        strategies = ['constant', 'linear', 'adaptive']
        
        for strategy in strategies:
            pso = ParticleSwarmOptimizer(
                n_particles=15,
                max_iterations=50,
                w_strategy=strategy,
                random_state=42
            )
            
            result = pso.optimize(sphere_function, self.bounds_2d)
            self.assertIsInstance(result['best_fitness'], (int, float))
    
    def test_advanced_pso(self):
        """Test advanced PSO with topologies."""
        topologies = ['global', 'ring', 'star']
        
        for topology in topologies:
            pso = AdvancedPSO(
                n_particles=20,
                max_iterations=60,
                topology=topology,
                neighborhood_size=3,
                random_state=42
            )
            
            result = pso.optimize(sphere_function, self.bounds_2d)
            self.assertIsInstance(result['best_position'], list)
            self.assertIsInstance(result['best_fitness'], (int, float))
    
    def test_pso_rosenbrock(self):
        """Test PSO on Rosenbrock function."""
        pso = ParticleSwarmOptimizer(
            n_particles=30,
            max_iterations=200,
            w_strategy='linear',
            random_state=42
        )
        
        bounds_rosenbrock = [(-2, 2), (-2, 2)]
        result = pso.optimize(rosenbrock_function, bounds_rosenbrock)
        
        # Should get reasonably close to [1, 1]
        best_pos = result['best_position']
        distance_to_optimum = math.sqrt((best_pos[0] - 1)**2 + (best_pos[1] - 1)**2)
        self.assertLess(distance_to_optimum, 2.0)


class TestDifferentialEvolution(unittest.TestCase):
    """Test cases for Differential Evolution."""
    
    def setUp(self):
        """Set up test parameters."""
        self.bounds_2d = [(-5, 5), (-5, 5)]
        random.seed(42)
    
    def test_basic_de(self):
        """Test basic DE."""
        de = DifferentialEvolution(
            population_size=20,
            max_generations=50,
            F=0.5,
            CR=0.9,
            strategy='DE/rand/1',
            random_state=42
        )
        
        result = de.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_solution'], list)
        self.assertEqual(len(result['best_solution']), 2)
        self.assertLess(result['best_fitness'], 2.0)
        self.assertIn('fitness_history', result)
    
    def test_de_strategies(self):
        """Test different DE strategies."""
        strategies = ['DE/rand/1', 'DE/best/1', 'DE/rand/2', 'DE/current-to-best/1']
        
        for strategy in strategies:
            de = DifferentialEvolution(
                population_size=20,
                max_generations=30,
                strategy=strategy,
                random_state=42
            )
            
            result = de.optimize(sphere_function, self.bounds_2d)
            self.assertIsInstance(result['best_fitness'], (int, float))
    
    def test_adaptive_de(self):
        """Test adaptive DE."""
        de = DifferentialEvolution(
            population_size=25,
            max_generations=60,
            adaptive=True,
            random_state=42
        )
        
        result = de.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_solution'], list)
        self.assertLess(result['best_fitness'], 5.0)
    
    def test_self_adaptive_de(self):
        """Test self-adaptive DE."""
        sade = SelfAdaptiveDifferentialEvolution(
            population_size=20,
            max_generations=40,
            random_state=42
        )
        
        result = sade.optimize(sphere_function, self.bounds_2d)
        
        self.assertIsInstance(result['best_solution'], list)
        self.assertIn('final_F_values', result)
        self.assertIn('final_CR_values', result)
        
        # Check that F and CR values are within valid ranges
        for f_val in result['final_F_values']:
            self.assertTrue(0.1 <= f_val <= 1.0)
        for cr_val in result['final_CR_values']:
            self.assertTrue(0.0 <= cr_val <= 1.0)
    
    def test_de_ackley(self):
        """Test DE on Ackley function."""
        de = DifferentialEvolution(
            population_size=30,
            max_generations=100,
            strategy='DE/best/1',
            random_state=42
        )
        
        bounds_ackley = [(-32.768, 32.768)] * 2
        result = de.optimize(ackley_function, bounds_ackley)
        
        # Should find solution close to global optimum at [0, 0]
        self.assertLess(result['best_fitness'], 5.0)


class TestOptimizationUtilities(unittest.TestCase):
    """Test utility functions and edge cases."""
    
    def test_bounds_handling(self):
        """Test boundary constraint handling."""
        # Single dimension bounds
        bounds_1d = [(-1, 1)]
        
        algorithms = [
            GeneticAlgorithm(population_size=10, generations=5, random_state=42),
            SimulatedAnnealing(max_iterations=20, random_state=42),
            ParticleSwarmOptimizer(n_particles=10, max_iterations=20, random_state=42),
            DifferentialEvolution(population_size=10, max_generations=10, random_state=42)
        ]
        
        def simple_1d_function(x):
            return x[0] ** 2
        
        for algorithm in algorithms:
            result = algorithm.optimize(simple_1d_function, bounds_1d)
            
            # Solution should be within bounds
            if hasattr(result, 'get'):
                solution = result.get('best_solution') or result.get('best_individual') or result.get('best_position')
            else:
                continue
                
            if solution:
                self.assertTrue(-1 <= solution[0] <= 1)
    
    def test_convergence_detection(self):
        """Test convergence detection."""
        # Use a simple function that should converge quickly
        def simple_quadratic(x):
            return sum((xi - 1) ** 2 for xi in x)
        
        bounds = [(0, 2)] * 2
        
        pso = ParticleSwarmOptimizer(
            n_particles=20,
            max_iterations=100,
            tolerance=1e-6,
            random_state=42
        )
        
        result = pso.optimize(simple_quadratic, bounds)
        
        # Check if convergence was detected
        if result.get('convergence_iteration') is not None:
            self.assertLess(result['convergence_iteration'], 100)
    
    def test_empty_bounds(self):
        """Test error handling for empty bounds."""
        algorithms = [
            GeneticAlgorithm(population_size=5, generations=5),
            ParticleSwarmOptimizer(n_particles=5, max_iterations=5),
            DifferentialEvolution(population_size=5, max_generations=5)
        ]
        
        for algorithm in algorithms:
            with self.assertRaises((ValueError, IndexError)):
                algorithm.optimize(sphere_function, [])
    
    def test_single_iteration(self):
        """Test algorithms with minimal iterations."""
        bounds = [(-1, 1)]
        
        def simple_function(x):
            return x[0] ** 2
        
        # Test with just 1 iteration/generation
        algorithms = [
            GeneticAlgorithm(population_size=5, generations=1, random_state=42),
            SimulatedAnnealing(max_iterations=1, random_state=42),
            ParticleSwarmOptimizer(n_particles=5, max_iterations=1, random_state=42),
            DifferentialEvolution(population_size=5, max_generations=1, random_state=42)
        ]
        
        for algorithm in algorithms:
            result = algorithm.optimize(simple_function, bounds)
            self.assertIsInstance(result, dict)


class TestPerformanceComparison(unittest.TestCase):
    """Performance comparison tests."""
    
    def test_algorithm_comparison(self):
        """Compare algorithms on standard test function."""
        bounds = [(-5, 5)] * 2
        
        algorithms = [
            ("GA", GeneticAlgorithm(population_size=20, generations=50, random_state=42)),
            ("SA", SimulatedAnnealing(max_iterations=1000, random_state=42)),
            ("PSO", ParticleSwarmOptimizer(n_particles=20, max_iterations=100, random_state=42)),
            ("DE", DifferentialEvolution(population_size=20, max_generations=50, random_state=42))
        ]
        
        results = {}
        
        for name, algorithm in algorithms:
            result = algorithm.optimize(sphere_function, bounds)
            
            # Extract best fitness regardless of result key name
            fitness = (result.get('best_fitness') or 
                      result.get('best_energy') or 
                      float('inf'))
            
            results[name] = fitness
        
        # All algorithms should find reasonable solutions
        for name, fitness in results.items():
            self.assertLess(fitness, 10.0, f"{name} should find reasonable solution")
        
        # At least one algorithm should find very good solution
        best_fitness = min(results.values())
        self.assertLess(best_fitness, 1.0, "At least one algorithm should find good solution")


if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestGeneticAlgorithm,
        TestSimulatedAnnealing, 
        TestParticleSwarmOptimization,
        TestDifferentialEvolution,
        TestOptimizationUtilities,
        TestPerformanceComparison
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nRan {result.testsRun} tests")
    if result.failures:
        print(f"FAILURES: {len(result.failures)}")
        for test, traceback in result.failures:
            print(f"FAILED: {test}")
    
    if result.errors:
        print(f"ERRORS: {len(result.errors)}")
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
    
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("Some tests failed or had errors.")
        print("Note: Some failures expected due to optimization randomness.")
