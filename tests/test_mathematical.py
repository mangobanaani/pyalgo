"""
Test cases for Mathematical algorithms
"""

import pytest
import math
from mathematical.number_theory import (
    gcd, lcm, is_prime, sieve_of_eratosthenes, prime_factorization,
    extended_gcd, mod_inverse, euler_totient, fast_power
)
from mathematical.linear_algebra import (
    Matrix, vector_dot, vector_cross, matrix_multiply, matrix_determinant,
    matrix_inverse, gaussian_elimination
)
from mathematical.numerical_methods import (
    newton_raphson, bisection_method, trapezoidal_rule, simpsons_rule,
    euler_method, runge_kutta_4, gradient_descent
)
from mathematical.statistics import (
    mean, median, mode, variance, standard_deviation, correlation,
    linear_regression, normal_distribution, percentile
)
from mathematical.combinatorics import (
    factorial, permutations, combinations, fibonacci, catalan_number,
    stirling_second_kind, bell_number, partition_function
)

class TestNumberTheory:
    def test_gcd(self):
        """Test greatest common divisor."""
        assert gcd(48, 18) == 6
        assert gcd(17, 13) == 1
        assert gcd(0, 5) == 5
        assert gcd(-12, 8) == 4
    
    def test_lcm(self):
        """Test least common multiple."""
        assert lcm(4, 6) == 12
        assert lcm(7, 5) == 35
        assert lcm(0, 5) == 0
    
    def test_is_prime(self):
        """Test primality testing."""
        assert is_prime(2)
        assert is_prime(17)
        assert is_prime(97)
        assert not is_prime(1)
        assert not is_prime(4)
        assert not is_prime(15)
        assert not is_prime(100)
    
    def test_sieve_of_eratosthenes(self):
        """Test prime sieve."""
        primes = sieve_of_eratosthenes(20)
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        assert primes == expected
        
        assert sieve_of_eratosthenes(1) == []
        assert sieve_of_eratosthenes(2) == [2]
    
    def test_prime_factorization(self):
        """Test prime factorization."""
        assert prime_factorization(12) == [(2, 2), (3, 1)]
        assert prime_factorization(17) == [(17, 1)]
        assert prime_factorization(100) == [(2, 2), (5, 2)]
        assert prime_factorization(1) == []
    
    def test_extended_gcd(self):
        """Test extended Euclidean algorithm."""
        gcd_val, x, y = extended_gcd(30, 20)
        assert gcd_val == 10
        assert 30 * x + 20 * y == gcd_val
    
    def test_mod_inverse(self):
        """Test modular inverse."""
        inv = mod_inverse(3, 7)
        assert inv == 5  # 3 * 5 ≡ 1 (mod 7)
        
        assert mod_inverse(2, 4) is None  # No inverse exists
    
    def test_euler_totient(self):
        """Test Euler's totient function."""
        assert euler_totient(1) == 1
        assert euler_totient(9) == 6  # φ(9) = 6
        assert euler_totient(10) == 4  # φ(10) = 4
    
    def test_fast_power(self):
        """Test fast exponentiation."""
        assert fast_power(2, 10) == 1024
        assert fast_power(3, 4, 5) == 1  # 3^4 mod 5 = 81 mod 5 = 1
        assert fast_power(5, 0) == 1

class TestLinearAlgebra:
    def test_matrix_creation(self):
        """Test matrix creation and access."""
        data = [[1, 2], [3, 4]]
        matrix = Matrix(data)
        
        assert matrix.rows == 2
        assert matrix.cols == 2
        assert matrix[0, 0] == 1
        assert matrix[1, 1] == 4
    
    def test_matrix_operations(self):
        """Test basic matrix operations."""
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[2, 0], [1, 2]])
        
        # Addition
        C = A + B
        assert C[0, 0] == 3
        assert C[1, 1] == 6
        
        # Scalar multiplication
        D = A * 2
        assert D[0, 0] == 2
        assert D[1, 1] == 8
    
    def test_vector_operations(self):
        """Test vector operations."""
        v1 = [1, 2, 3]
        v2 = [4, 5, 6]
        
        # Dot product
        assert vector_dot(v1, v2) == 32
        
        # Cross product
        cross = vector_cross(v1, v2)
        assert cross == [-3, 6, -3]
    
    def test_matrix_multiply(self):
        """Test matrix multiplication."""
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[2, 0], [1, 2]])
        
        C = matrix_multiply(A, B)
        assert C[0, 0] == 4
        assert C[0, 1] == 4
        assert C[1, 0] == 10
        assert C[1, 1] == 8
    
    def test_matrix_determinant(self):
        """Test matrix determinant."""
        A = Matrix([[1, 2], [3, 4]])
        det = matrix_determinant(A)
        assert abs(det - (-2)) < 1e-10
        
        # Identity matrix
        I = Matrix.identity(3)
        assert abs(matrix_determinant(I) - 1) < 1e-10
    
    def test_gaussian_elimination(self):
        """Test solving linear systems."""
        A = Matrix([[2, 1], [1, 1]])
        b = [3, 2]
        
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 1) < 1e-10
        assert abs(x[1] - 1) < 1e-10

class TestNumericalMethods:
    def test_newton_raphson(self):
        """Test Newton-Raphson root finding."""
        # Find root of x^2 - 2 = 0 (should be √2)
        def f(x):
            return x*x - 2
        
        def df(x):
            return 2*x
        
        root = newton_raphson(f, df, 1.0)
        assert abs(root - math.sqrt(2)) < 1e-6
    
    def test_bisection_method(self):
        """Test bisection method."""
        def f(x):
            return x*x - 2
        
        root = bisection_method(f, 1, 2)
        assert abs(root - math.sqrt(2)) < 1e-6
    
    def test_trapezoidal_rule(self):
        """Test numerical integration."""
        # Integrate x^2 from 0 to 1 (should be 1/3)
        def f(x):
            return x*x
        
        integral = trapezoidal_rule(f, 0, 1, 1000)
        assert abs(integral - 1/3) < 1e-2
    
    def test_simpsons_rule(self):
        """Test Simpson's rule integration."""
        def f(x):
            return x*x
        
        integral = simpsons_rule(f, 0, 1, 1000)
        assert abs(integral - 1/3) < 1e-6
    
    def test_euler_method(self):
        """Test Euler's method for ODEs."""
        # Solve dy/dt = y with y(0) = 1 (solution is e^t)
        def f(t, y):
            return y
        
        points = euler_method(f, 1.0, 0.0, 1.0, 0.1)
        
        # Check that solution is approximately e^1 at t=1
        final_y = points[-1][1]
        assert abs(final_y - math.e) < 0.5  # Euler method is not very accurate
    
    def test_runge_kutta_4(self):
        """Test 4th-order Runge-Kutta method."""
        def f(t, y):
            return y
        
        points = runge_kutta_4(f, 1.0, 0.0, 1.0, 0.1)
        
        # Should be much more accurate than Euler
        final_y = points[-1][1]
        assert abs(final_y - math.e) < 1e-3

class TestStatistics:
    def test_mean(self):
        """Test arithmetic mean."""
        data = [1, 2, 3, 4, 5]
        assert mean(data) == 3.0
        
        data = [2.5, 3.5, 4.5]
        assert abs(mean(data) - 3.5) < 1e-10
    
    def test_median(self):
        """Test median calculation."""
        # Odd number of elements
        data = [1, 2, 3, 4, 5]
        assert median(data) == 3.0
        
        # Even number of elements
        data = [1, 2, 3, 4]
        assert median(data) == 2.5
    
    def test_mode(self):
        """Test mode calculation."""
        data = [1, 2, 2, 3, 4]
        modes = mode(data)
        assert modes == [2]
        
        # Multiple modes
        data = [1, 1, 2, 2, 3]
        modes = mode(data)
        assert set(modes) == {1, 2}
    
    def test_variance_and_std(self):
        """Test variance and standard deviation."""
        data = [1, 2, 3, 4, 5]
        
        var = variance(data, population=True)
        assert abs(var - 2.0) < 1e-10
        
        std = standard_deviation(data, population=True)
        assert abs(std - math.sqrt(2)) < 1e-10
    
    def test_correlation(self):
        """Test correlation coefficient."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Perfect positive correlation
        
        corr = correlation(x, y)
        assert abs(corr - 1.0) < 1e-10
        
        # Perfect negative correlation
        y = [10, 8, 6, 4, 2]
        corr = correlation(x, y)
        assert abs(corr - (-1.0)) < 1e-10
    
    def test_linear_regression(self):
        """Test linear regression."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # y = 2x
        
        slope, intercept, r_squared = linear_regression(x, y)
        assert abs(slope - 2.0) < 1e-10
        assert abs(intercept - 0.0) < 1e-10
        assert abs(r_squared - 1.0) < 1e-10
    
    def test_normal_distribution(self):
        """Test normal distribution PDF."""
        # Standard normal at x=0 should be 1/√(2π)
        pdf = normal_distribution(0, 0, 1)
        expected = 1 / math.sqrt(2 * math.pi)
        assert abs(pdf - expected) < 1e-10
    
    def test_percentile(self):
        """Test percentile calculation."""
        data = list(range(1, 101))  # 1 to 100
        
        assert percentile(data, 50) == 50.5  # Median
        assert abs(percentile(data, 25) - 25.75) < 1e-10  # First quartile
        assert abs(percentile(data, 75) - 75.25) < 1e-10  # Third quartile

class TestCombinatorics:
    def test_factorial(self):
        """Test factorial calculation."""
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(6) == 720
    
    def test_permutations(self):
        """Test permutation calculation."""
        assert permutations(5, 3) == 60  # 5*4*3
        assert permutations(5, 5) == 120  # 5!
        assert permutations(5, 0) == 1
        assert permutations(3, 5) == 0  # r > n
    
    def test_combinations(self):
        """Test combination calculation."""
        assert combinations(5, 3) == 10
        assert combinations(5, 0) == 1
        assert combinations(5, 5) == 1
        assert combinations(3, 5) == 0  # r > n
    
    def test_fibonacci(self):
        """Test Fibonacci sequence."""
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        for i, expected_val in enumerate(expected):
            assert fibonacci(i) == expected_val
    
    def test_catalan_number(self):
        """Test Catalan numbers."""
        # First few Catalan numbers: 1, 1, 2, 5, 14, 42
        expected = [1, 1, 2, 5, 14, 42]
        for i, expected_val in enumerate(expected):
            assert catalan_number(i) == expected_val
    
    def test_stirling_second_kind(self):
        """Test Stirling numbers of second kind."""
        assert stirling_second_kind(4, 2) == 7
        assert stirling_second_kind(5, 3) == 25
        assert stirling_second_kind(3, 3) == 1
        assert stirling_second_kind(3, 4) == 0  # k > n
    
    def test_bell_number(self):
        """Test Bell numbers."""
        # First few Bell numbers: 1, 1, 2, 5, 15
        expected = [1, 1, 2, 5, 15]
        for i, expected_val in enumerate(expected):
            assert bell_number(i) == expected_val
    
    def test_partition_function(self):
        """Test integer partition function."""
        assert partition_function(4) == 5  # 4, 3+1, 2+2, 2+1+1, 1+1+1+1
        assert partition_function(5) == 7
        assert partition_function(0) == 1

class TestIntegration:
    def test_mathematical_integration(self):
        """Test integration between different mathematical modules."""
        # Use linear algebra to solve a system, then do statistics on result
        from mathematical.linear_algebra import Matrix, gaussian_elimination
        from mathematical.statistics import mean, variance
        
        # Solve multiple systems and analyze results
        results = []
        for i in range(1, 6):
            A = Matrix([[2, 1], [1, 1]])
            b = [i + 2, i + 1]
            x = gaussian_elimination(A, b)
            results.append(x[1])  # Second component of solution (varies 1,2,3,4,5)
        
        # Statistical analysis - results should be [1, 2, 3, 4, 5]
        result_mean = mean(results)
        result_var = variance(results, population=True)
        
        assert abs(result_mean - 3.0) < 1e-10
        assert abs(result_var - 2.0) < 1e-10
    
    def test_numerical_and_combinatorial(self):
        """Test combination of numerical methods and combinatorics."""
        from mathematical.numerical_methods import trapezoidal_rule
        from mathematical.combinatorics import factorial
        
        # Integrate factorial approximation (Stirling's approximation related)
        def stirling_approx(x):
            if x < 1:
                return 1
            return math.sqrt(2 * math.pi * x) * (x / math.e) ** x
        
        # Compare with actual factorial for small values
        for n in range(1, 6):
            actual = factorial(n)
            approx = stirling_approx(n)
            error = abs(actual - approx) / actual
            
            # Stirling approximation gets better for larger n
            if n >= 3:
                assert error < 0.2  # Within 20% for n >= 3
