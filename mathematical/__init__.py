"""
Mathematical algorithms package.

This package contains implementations of various mathematical algorithms including:
- Number theory algorithms
- Linear algebra operations
- Calculus and numerical methods
- Probability and statistics
- Combinatorics
- Matrix operations
"""

from .number_theory import (
    gcd, lcm, is_prime, sieve_of_eratosthenes, prime_factorization,
    extended_gcd, mod_inverse, euler_totient, fast_power
)
from .linear_algebra import (
    Matrix, vector_dot, vector_cross, matrix_multiply, matrix_determinant,
    matrix_inverse, matrix_transpose, gaussian_elimination, lu_decomposition
)
from .numerical_methods import (
    newton_raphson, bisection_method, trapezoidal_rule, simpsons_rule,
    euler_method, runge_kutta_4, gradient_descent
)
from .statistics import (
    mean, median, mode, variance, standard_deviation, correlation,
    linear_regression, normal_distribution, chi_square_test
)
from .combinatorics import (
    factorial, permutations, combinations, stirling_second_kind,
    catalan_number, fibonacci, lucas_numbers, partition_function
)

__all__ = [
    # Number theory
    'gcd', 'lcm', 'is_prime', 'sieve_of_eratosthenes', 'prime_factorization',
    'extended_gcd', 'mod_inverse', 'euler_totient', 'fast_power',
    
    # Linear algebra
    'Matrix', 'vector_dot', 'vector_cross', 'matrix_multiply', 'matrix_determinant',
    'matrix_inverse', 'matrix_transpose', 'gaussian_elimination', 'lu_decomposition',
    
    # Numerical methods
    'newton_raphson', 'bisection_method', 'trapezoidal_rule', 'simpsons_rule',
    'euler_method', 'runge_kutta_4', 'gradient_descent',
    
    # Statistics
    'mean', 'median', 'mode', 'variance', 'standard_deviation', 'correlation',
    'linear_regression', 'normal_distribution', 'chi_square_test',
    
    # Combinatorics
    'factorial', 'permutations', 'combinations', 'stirling_second_kind',
    'catalan_number', 'fibonacci', 'lucas_numbers', 'partition_function'
]
