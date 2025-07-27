"""
Numerical Methods for solving mathematical problems.

This module implements numerical algorithms for root finding, integration,
differential equations, and optimization.
"""

from typing import Callable, List, Tuple, Optional
import math


def newton_raphson(f: Callable[[float], float], 
                  df: Callable[[float], float],
                  x0: float, 
                  tolerance: float = 1e-6,
                  max_iterations: int = 100) -> Optional[float]:
    """
    Find root of function using Newton-Raphson method.
    
    Args:
        f: Function to find root of
        df: Derivative of function
        x0: Initial guess
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Root if found, None if method doesn't converge
        
    Time Complexity: O(max_iterations)
    Space Complexity: O(1)
    """
    x = x0
    
    for _ in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        
        if abs(fx) < tolerance:
            return x
        
        if abs(dfx) < 1e-12:
            return None  # Derivative too small, might not converge
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tolerance:
            return x_new
        
        x = x_new
    
    return None  # Didn't converge


def bisection_method(f: Callable[[float], float],
                    a: float, b: float,
                    tolerance: float = 1e-6,
                    max_iterations: int = 100) -> Optional[float]:
    """
    Find root using bisection method.
    
    Args:
        f: Continuous function
        a: Left endpoint (f(a) and f(b) must have opposite signs)
        b: Right endpoint
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Root if found, None if invalid interval
        
    Time Complexity: O(max_iterations)
    Space Complexity: O(1)
    """
    fa, fb = f(a), f(b)
    
    # Check if root exists in interval
    if fa * fb > 0:
        return None  # No root in interval
    
    if abs(fa) < tolerance:
        return a
    if abs(fb) < tolerance:
        return b
    
    for _ in range(max_iterations):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tolerance or abs(b - a) < tolerance:
            return c
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return (a + b) / 2


def secant_method(f: Callable[[float], float],
                 x0: float, x1: float,
                 tolerance: float = 1e-6,
                 max_iterations: int = 100) -> Optional[float]:
    """
    Find root using secant method.
    
    Args:
        f: Function to find root of
        x0: First initial point
        x1: Second initial point
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Root if found, None if method doesn't converge
        
    Time Complexity: O(max_iterations)
    Space Complexity: O(1)
    """
    for _ in range(max_iterations):
        f0, f1 = f(x0), f(x1)
        
        if abs(f1) < tolerance:
            return x1
        
        if abs(f1 - f0) < 1e-12:
            return None  # Division by zero
        
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        if abs(x2 - x1) < tolerance:
            return x2
        
        x0, x1 = x1, x2
    
    return None


def trapezoidal_rule(f: Callable[[float], float],
                    a: float, b: float,
                    n: int = 1000) -> float:
    """
    Numerical integration using trapezoidal rule.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of subdivisions
        
    Returns:
        Approximate integral value
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 0:
        raise ValueError("Number of subdivisions must be positive")
    
    h = (b - a) / n
    result = (f(a) + f(b)) / 2
    
    for i in range(1, n):
        x = a + i * h
        result += f(x)
    
    return result * h


def simpsons_rule(f: Callable[[float], float],
                 a: float, b: float,
                 n: int = 1000) -> float:
    """
    Numerical integration using Simpson's rule.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n: Number of subdivisions (must be even)
        
    Returns:
        Approximate integral value
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError("Number of subdivisions must be positive and even")
    
    h = (b - a) / n
    result = f(a) + f(b)
    
    # Add odd-indexed terms (coefficient 4)
    for i in range(1, n, 2):
        x = a + i * h
        result += 4 * f(x)
    
    # Add even-indexed terms (coefficient 2)
    for i in range(2, n, 2):
        x = a + i * h
        result += 2 * f(x)
    
    return result * h / 3


def romberg_integration(f: Callable[[float], float],
                       a: float, b: float,
                       max_steps: int = 10,
                       tolerance: float = 1e-6) -> float:
    """
    Numerical integration using Romberg's method.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        max_steps: Maximum number of refinement steps
        tolerance: Convergence tolerance
        
    Returns:
        Approximate integral value
        
    Time Complexity: O(4^max_steps)
    Space Complexity: O(max_steps²)
    """
    # Initialize Romberg table
    R = [[0.0 for _ in range(max_steps)] for _ in range(max_steps)]
    
    h = b - a
    R[0][0] = h * (f(a) + f(b)) / 2
    
    for i in range(1, max_steps):
        # Calculate R[i][0] using trapezoidal rule with 2^i intervals
        h /= 2
        sum_new = 0
        for k in range(1, 2**i, 2):
            sum_new += f(a + k * h)
        
        R[i][0] = R[i-1][0] / 2 + h * sum_new
        
        # Calculate R[i][j] using Richardson extrapolation
        for j in range(1, i + 1):
            power_4 = 4**j
            R[i][j] = (power_4 * R[i][j-1] - R[i-1][j-1]) / (power_4 - 1)
        
        # Check convergence
        if i > 0 and abs(R[i][i] - R[i-1][i-1]) < tolerance:
            return R[i][i]
    
    return R[max_steps-1][max_steps-1]


def euler_method(f: Callable[[float, float], float],
                y0: float, t0: float, t_final: float,
                h: float) -> List[Tuple[float, float]]:
    """
    Solve ODE dy/dt = f(t,y) using Euler's method.
    
    Args:
        f: Function f(t,y) defining the ODE
        y0: Initial condition y(t0) = y0
        t0: Initial time
        t_final: Final time
        h: Step size
        
    Returns:
        List of (t, y) points
        
    Time Complexity: O(n) where n = (t_final - t0) / h
    Space Complexity: O(n)
    """
    if h <= 0:
        raise ValueError("Step size must be positive")
    
    points = [(t0, y0)]
    t, y = t0, y0
    
    while t < t_final:
        if t + h > t_final:
            h = t_final - t  # Adjust last step
        
        y += h * f(t, y)
        t += h
        points.append((t, y))
    
    return points


def runge_kutta_4(f: Callable[[float, float], float],
                 y0: float, t0: float, t_final: float,
                 h: float) -> List[Tuple[float, float]]:
    """
    Solve ODE dy/dt = f(t,y) using 4th-order Runge-Kutta method.
    
    Args:
        f: Function f(t,y) defining the ODE
        y0: Initial condition y(t0) = y0
        t0: Initial time
        t_final: Final time
        h: Step size
        
    Returns:
        List of (t, y) points
        
    Time Complexity: O(n) where n = (t_final - t0) / h
    Space Complexity: O(n)
    """
    if h <= 0:
        raise ValueError("Step size must be positive")
    
    points = [(t0, y0)]
    t, y = t0, y0
    
    while t < t_final:
        if t + h > t_final:
            h = t_final - t
        
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        points.append((t, y))
    
    return points


def gradient_descent(f: Callable[[List[float]], float],
                    grad_f: Callable[[List[float]], List[float]],
                    x0: List[float],
                    learning_rate: float = 0.01,
                    tolerance: float = 1e-6,
                    max_iterations: int = 1000) -> List[float]:
    """
    Find minimum of function using gradient descent.
    
    Args:
        f: Function to minimize
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Step size
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Point of minimum
        
    Time Complexity: O(max_iterations * d) where d is dimension
    Space Complexity: O(d)
    """
    x = x0[:]
    
    for _ in range(max_iterations):
        grad = grad_f(x)
        
        # Check convergence
        grad_norm = math.sqrt(sum(g*g for g in grad))
        if grad_norm < tolerance:
            break
        
        # Update x
        for i in range(len(x)):
            x[i] -= learning_rate * grad[i]
    
    return x


def newton_optimization(f: Callable[[List[float]], float],
                       grad_f: Callable[[List[float]], List[float]],
                       hess_f: Callable[[List[float]], List[List[float]]],
                       x0: List[float],
                       tolerance: float = 1e-6,
                       max_iterations: int = 100) -> List[float]:
    """
    Find minimum using Newton's method for optimization.
    
    Args:
        f: Function to minimize
        grad_f: Gradient function
        hess_f: Hessian function
        x0: Initial point
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Point of minimum
        
    Time Complexity: O(max_iterations * d³) where d is dimension
    Space Complexity: O(d²)
    """
    from .linear_algebra import Matrix, gaussian_elimination
    
    x = x0[:]
    
    for _ in range(max_iterations):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Check convergence
        grad_norm = math.sqrt(sum(g*g for g in grad))
        if grad_norm < tolerance:
            break
        
        # Solve Hessian * delta = -gradient
        hess_matrix = Matrix(hess)
        neg_grad = [-g for g in grad]
        
        delta = gaussian_elimination(hess_matrix, neg_grad)
        if delta is None:
            # Hessian is singular, fall back to gradient descent
            delta = [-0.01 * g for g in grad]
        
        # Update x
        for i in range(len(x)):
            x[i] += delta[i]
    
    return x


def finite_difference_derivative(f: Callable[[float], float],
                                x: float,
                                h: float = 1e-5) -> float:
    """
    Approximate derivative using finite differences.
    
    Args:
        f: Function to differentiate
        x: Point to evaluate derivative
        h: Step size
        
    Returns:
        Approximate derivative
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def finite_difference_gradient(f: Callable[[List[float]], float],
                              x: List[float],
                              h: float = 1e-5) -> List[float]:
    """
    Approximate gradient using finite differences.
    
    Args:
        f: Multivariable function
        x: Point to evaluate gradient
        h: Step size
        
    Returns:
        Approximate gradient vector
        
    Time Complexity: O(d) where d is dimension
    Space Complexity: O(d)
    """
    grad = []
    
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        
        partial = (f(x_plus) - f(x_minus)) / (2 * h)
        grad.append(partial)
    
    return grad
