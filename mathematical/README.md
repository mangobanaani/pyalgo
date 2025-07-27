# Mathematical Algorithms

Comprehensive collection of mathematical algorithms covering number theory, linear algebra, numerical methods, statistics, and combinatorics.

## Modules Included

### Number Theory (`number_theory.py`)
Core algorithms for working with integers and prime numbers:
- **GCD and LCM**: Euclidean algorithm for greatest common divisor and least common multiple
- **Prime Testing**: Efficient primality testing and Sieve of Eratosthenes
- **Modular Arithmetic**: Fast modular exponentiation, modular inverse, Chinese Remainder Theorem
- **Extended GCD**: Bézout coefficients and linear Diophantine equations
- **Euler's Totient**: Count of integers coprime to n

### Linear Algebra (`linear_algebra.py`)
Matrix operations and linear system solving:
- **Matrix Class**: Complete implementation with arithmetic operations
- **Matrix Operations**: Addition, multiplication, determinant, inverse, transpose
- **Vector Operations**: Dot product, cross product, vector norms
- **System Solving**: Gaussian elimination, LU decomposition, QR decomposition
- **Eigenvalues**: Power iteration method for dominant eigenvalue

### Numerical Methods (`numerical_methods.py`)
Algorithms for numerical analysis and computational mathematics:
- **Root Finding**: Newton-Raphson method, bisection method
- **Integration**: Trapezoidal rule, Simpson's rule for numerical integration
- **Differential Equations**: Euler's method, Runge-Kutta 4th order method
- **Optimization**: Gradient descent for function minimization
- **Interpolation**: Lagrange polynomial interpolation

### Statistics (`statistics.py`)
Statistical analysis and probability distributions:
- **Descriptive Statistics**: Mean, median, mode, variance, standard deviation
- **Correlation Analysis**: Pearson correlation coefficient
- **Regression**: Linear regression with R-squared calculation
- **Probability Distributions**: Normal, binomial, chi-square distributions
- **Hypothesis Testing**: Chi-square test, t-test functions
- **Percentiles**: Percentile calculation with linear interpolation

### Combinatorics (`combinatorics.py`)
Counting and sequence algorithms:
- **Basic Counting**: Factorials, permutations, combinations
- **Special Sequences**: Fibonacci numbers, Catalan numbers
- **Advanced Counting**: Stirling numbers (first and second kind), Bell numbers
- **Partition Functions**: Integer partitioning algorithms
- **Generating Functions**: Implementations for various combinatorial sequences

### Legacy Modules
- **`sieve_of_eratosthenes.py`**: Standalone prime number generation
- **`gcd.py`**: Basic GCD implementation
- **`modular_arithmetic.py`**: Modular operations

## Usage Examples

### Number Theory
```python
from mathematical.number_theory import gcd, is_prime, fast_power, chinese_remainder_theorem

# Basic number theory
print(gcd(48, 18))  # 6
print(is_prime(17))  # True
print(fast_power(2, 10))  # 1024
print(fast_power(3, 4, 5))  # 1 (3^4 mod 5)

# Chinese Remainder Theorem
remainders = [2, 3, 2]
moduli = [3, 5, 7]
result = chinese_remainder_theorem(remainders, moduli)
print(f"CRT solution: {result}")
```

### Linear Algebra
```python
from mathematical.linear_algebra import Matrix, gaussian_elimination

# Matrix operations
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[2, 0], [1, 2]])
C = A + B  # Matrix addition
D = A * B  # Matrix multiplication

print(f"Determinant of A: {A.determinant()}")

# Solve linear system Ax = b
A = Matrix([[2, 1], [1, 1]])
b = [3, 2]
solution = gaussian_elimination(A, b)
print(f"Solution: x = {solution}")
```

### Numerical Methods
```python
from mathematical.numerical_methods import newton_raphson, trapezoidal_rule

# Find root of x^2 - 2 = 0 (should be √2)
def f(x):
    return x*x - 2

def df(x):
    return 2*x

root = newton_raphson(f, df, 1.0)
print(f"Square root of 2: {root}")

# Numerical integration of x^2 from 0 to 1
def integrand(x):
    return x*x

integral = trapezoidal_rule(integrand, 0, 1, 1000)
print(f"Integral of x^2 from 0 to 1: {integral}")  # Should be ≈ 1/3
```

### Statistics
```python
from mathematical.statistics import mean, variance, correlation, linear_regression

# Basic statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Mean: {mean(data)}")
print(f"Variance: {variance(data)}")

# Correlation and regression
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]  # Perfect linear relationship
corr = correlation(x, y)
slope, intercept, r_squared = linear_regression(x, y)
print(f"Correlation: {corr}")
print(f"Regression: y = {slope}x + {intercept}, R² = {r_squared}")
```

### Combinatorics
```python
from mathematical.combinatorics import factorial, combinations, fibonacci, catalan_number

# Basic combinatorics
print(f"5! = {factorial(5)}")
print(f"C(10, 3) = {combinations(10, 3)}")
print(f"10th Fibonacci: {fibonacci(10)}")
print(f"5th Catalan: {catalan_number(5)}")

# Generate Fibonacci sequence
fib_sequence = [fibonacci(i) for i in range(10)]
print(f"Fibonacci sequence: {fib_sequence}")
```

## Algorithm Complexity

### Number Theory
- **GCD**: O(log(min(a,b)))
- **Prime Testing**: O(√n) for trial division
- **Sieve of Eratosthenes**: O(n log log n)
- **Fast Power**: O(log n)

### Linear Algebra
- **Matrix Multiplication**: O(n³)
- **Determinant**: O(n³)
- **Gaussian Elimination**: O(n³)
- **LU Decomposition**: O(n³)

### Numerical Methods
- **Newton-Raphson**: O(k) where k is iterations to convergence
- **Trapezoidal Rule**: O(n) where n is number of intervals
- **Runge-Kutta**: O(k) where k is number of steps

### Statistics
- **Mean/Variance**: O(n)
- **Correlation**: O(n)
- **Linear Regression**: O(n)
- **Percentile**: O(n log n) due to sorting

### Combinatorics
- **Factorial**: O(n)
- **Combinations**: O(min(k, n-k))
- **Fibonacci**: O(n) with memoization
- **Catalan Numbers**: O(n)

## Applications

- **Cryptography**: RSA key generation, modular arithmetic
- **Computer Graphics**: Matrix transformations, geometric algorithms
- **Data Science**: Statistical analysis, regression modeling
- **Scientific Computing**: Numerical simulation, equation solving
- **Algorithm Analysis**: Complexity calculations, generating functions
